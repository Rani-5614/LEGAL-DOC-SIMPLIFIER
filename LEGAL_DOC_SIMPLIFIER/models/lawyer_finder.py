# models/lawyer_finder.py
"""
Location-based lawyer discovery.

Data sources (no API key needed):
  * OpenStreetMap Overpass API  -> real lawyer / advocate / notary offices around the user
  * OpenStreetMap Nominatim     -> reverse geocoding (human-readable address of the user)
  * Local `LegalExpert` table   -> verified/curated lawyers, now ranked by real distance

OpenStreetMap rarely records a lawyer's practice area. So specialization is
  - "listed" when the OSM entry (name/tags/description) mentions the required area, and
  - otherwise shown as "General practice / verify specialization" - we never invent one.
"""
import math
import re
import requests

OVERPASS_URLS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.private.coffee/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
]
NOMINATIM_SEARCH = 'https://nominatim.openstreetmap.org/search'
GOOGLE_PLACES_URL = 'https://places.googleapis.com/v1/places:searchText'
NOMINATIM_REVERSE = 'https://nominatim.openstreetmap.org/reverse'
HEADERS = {'User-Agent': 'LegalEase/2.0 (legal document simplifier)'}

# Keywords that hint a practice area in names / tags / descriptions
SPECIALIZATION_KEYWORDS = {
    'Contract & Business Law': ['contract', 'business', 'commercial', 'corporate'],
    'Employment & Labour Law': ['employment', 'labour', 'labor', 'service matter', 'industrial'],
    'Property & Real Estate Law': ['property', 'real estate', 'land', 'conveyanc', 'revenue', 'title'],
    'Banking & Finance Law': ['bank', 'finance', 'debt', 'recovery', 'cheque', 'tax', 'drt'],
    'Family Law': ['family', 'divorce', 'matrimon', 'marriage', 'custody', 'succession', 'will'],
    'Corporate & Company Law': ['corporate', 'company', 'nclt', 'securities', 'roc'],
    'Litigation & Administrative Law': ['litigation', 'civil', 'writ', 'constitutional', 'administrative', 'high court'],
    'Intellectual Property Law': ['patent', 'trademark', 'copyright', 'intellectual', 'ip '],
    'Criminal Law': ['criminal', 'bail', 'sessions', 'defence', 'defense'],
    'Civil Litigation': ['civil', 'litigation', 'consumer', 'tort'],
    'Insurance Law': ['insurance', 'claim', 'consumer'],
    'Education Law': ['education', 'university', 'service matter'],
    'Medical & Healthcare Law': ['medical', 'health', 'negligence', 'consumer'],
    'General Legal Practice': ['advocate', 'lawyer', 'legal', 'associates', 'law'],
}


def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def reverse_geocode(lat, lon):
    """Return a short readable place name for coordinates, or None."""
    try:
        r = requests.get(NOMINATIM_REVERSE, headers=HEADERS, timeout=8,
                         params={'lat': lat, 'lon': lon, 'format': 'jsonv2', 'zoom': 14})
        if r.ok:
            data = r.json()
            a = data.get('address', {})
            parts = [a.get('suburb') or a.get('neighbourhood') or a.get('village'),
                     a.get('city') or a.get('town') or a.get('county'),
                     a.get('state')]
            name = ', '.join(p for p in parts if p)
            return name or data.get('display_name')
    except Exception as e:
        print(f'[LawyerFinder] reverse geocode failed: {e}')
    return None


def reverse_geocode_full(lat, lon):
    """Return {'label': short place, 'address': full street address} for the device location."""
    out = {'label': None, 'address': None}
    try:
        r = requests.get(NOMINATIM_REVERSE, headers=HEADERS, timeout=8,
                         params={'lat': lat, 'lon': lon, 'format': 'jsonv2', 'zoom': 18, 'addressdetails': 1})
        if r.ok:
            data = r.json()
            a = data.get('address', {})
            parts = [a.get('suburb') or a.get('neighbourhood') or a.get('village'),
                     a.get('city') or a.get('town') or a.get('county'), a.get('state')]
            out['label'] = ', '.join(p for p in parts if p) or None
            out['address'] = data.get('display_name')
    except Exception as e:
        print(f'[LawyerFinder] reverse geocode failed: {e}')
    return out


def directions_urls(dest_lat, dest_lon, origin_lat=None, origin_lon=None, name=None):
    q = f'{dest_lat},{dest_lon}'
    google = f'https://www.google.com/maps/dir/?api=1&destination={q}&travelmode=driving'
    if origin_lat is not None and origin_lon is not None:
        google += f'&origin={origin_lat},{origin_lon}'
    osm = (f'https://www.openstreetmap.org/directions?engine=fossgis_osrm_car'
           f'&route={origin_lat},{origin_lon}%3B{dest_lat},{dest_lon}'
           if origin_lat is not None else f'https://www.openstreetmap.org/?mlat={dest_lat}&mlon={dest_lon}#map=17/{dest_lat}/{dest_lon}')
    return {'google': google, 'osm': osm}


def _format_address(tags):
    parts = []
    street = ' '.join(x for x in [tags.get('addr:housenumber'), tags.get('addr:street')] if x)
    for p in [tags.get('addr:housename'), street, tags.get('addr:suburb'),
              tags.get('addr:city') or tags.get('addr:town') or tags.get('addr:village'),
              tags.get('addr:state'), tags.get('addr:postcode')]:
        if p:
            parts.append(p)
    return ', '.join(parts) if parts else (tags.get('addr:full') or '')


def _infer_specialization(tags, primary, related):
    """Returns (label, matched_bool)."""
    blob = ' '.join(str(tags.get(k, '')) for k in
                    ('name', 'description', 'lawyer', 'speciality', 'specialty', 'operator', 'brand')).lower()
    wanted = [primary] + list(related)
    for area in wanted:
        for kw in SPECIALIZATION_KEYWORDS.get(area, []):
            if kw in ('advocate', 'lawyer', 'legal', 'associates', 'law'):
                continue
            if kw in blob:
                return area, True
    return 'General practice (confirm ' + primary + ' expertise)', False


def _overpass_query(lat, lon, radius_m):
    # Tag-only query (no slow regex scans) so it finishes quickly even in dense cities.
    return (f"[out:json][timeout:20];("
            f'nwr["office"="lawyer"](around:{radius_m},{lat},{lon});'
            f'nwr["amenity"="lawyer"](around:{radius_m},{lat},{lon});'
            f'nwr["office"="notary"](around:{radius_m},{lat},{lon});'
            f'nwr["office"="advocate"](around:{radius_m},{lat},{lon});'
            f");out center tags 80;")


def _normalize(source, ident, name, el_lat, el_lon, tags_like, user_lat, user_lon, primary, related,
               address='', phone=None, email=None, website=None, hours=None, rating=None):
    spec, matched = _infer_specialization(tags_like, primary, related)
    dist = haversine_km(user_lat, user_lon, el_lat, el_lon)
    urls = directions_urls(el_lat, el_lon, user_lat, user_lon, name)
    return {
        'id': ident, 'source': source, 'name': name,
        'specialization': spec, 'specialization_matched': matched,
        'address': address or 'Address not listed - use directions',
        'phone': phone, 'email': email, 'website': website, 'opening_hours': hours,
        'latitude': el_lat, 'longitude': el_lon,
        'distance_km': round(dist, 2) if dist is not None else None,
        'directions_google': urls['google'], 'directions_osm': urls['osm'],
        'rating': rating, 'experience': None, 'fee_range': None,
    }


def fetch_osm_lawyers(lat, lon, radius_km=10, primary='General Legal Practice', related=(), errors=None):
    """Overpass lookup across several mirrors. Returns list, or None if every mirror failed."""
    query = _overpass_query(lat, lon, int(radius_km * 1000))
    data = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={'data': query}, timeout=25,
                              headers={**HEADERS, 'Accept': 'application/json'})
            if r.ok and r.text.lstrip().startswith('{'):
                data = r.json()
                break
            if errors is not None:
                errors.append(f'Overpass {url.split("/")[2]}: HTTP {r.status_code}')
        except Exception as e:
            if errors is not None:
                errors.append(f'Overpass {url.split("/")[2]}: {type(e).__name__}')
            print(f'[LawyerFinder] Overpass {url} failed: {e}')
    if data is None:
        return None
    results, seen = [], set()
    for el in data.get('elements', []):
        tags = el.get('tags', {})
        name = tags.get('name') or tags.get('operator')
        el_lat = el.get('lat') or (el.get('center') or {}).get('lat')
        el_lon = el.get('lon') or (el.get('center') or {}).get('lon')
        if not name or el_lat is None or el_lon is None:
            continue
        key = (name.lower(), round(el_lat, 4), round(el_lon, 4))
        if key in seen:
            continue
        seen.add(key)
        results.append(_normalize(
            'OpenStreetMap', f"osm-{el.get('type', 'n')}-{el.get('id')}", name, el_lat, el_lon, tags,
            lat, lon, primary, related, address=_format_address(tags),
            phone=tags.get('phone') or tags.get('contact:phone') or tags.get('contact:mobile'),
            email=tags.get('email') or tags.get('contact:email'),
            website=tags.get('website') or tags.get('contact:website'),
            hours=tags.get('opening_hours')))
    return results


def fetch_nominatim_lawyers(lat, lon, radius_km=10, primary='General Legal Practice', related=(), errors=None):
    """
    Second source: Nominatim free-text search ("advocate", "lawyer"...) restricted to a box around the user.
    Independent of Overpass, so it still works when Overpass is overloaded.
    """
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    viewbox = f'{lon - deg_lon},{lat + deg_lat},{lon + deg_lon},{lat - deg_lat}'
    terms = ['advocate', 'lawyer', 'law office', 'legal services']
    if primary and primary != 'General Legal Practice':
        terms.insert(0, primary.split('&')[0].strip() + ' lawyer')
    results, seen, ok = [], set(), False
    for term in terms:
        try:
            r = requests.get(NOMINATIM_SEARCH, headers=HEADERS, timeout=10, params={
                'q': term, 'format': 'jsonv2', 'viewbox': viewbox, 'bounded': 1, 'limit': 30,
                'addressdetails': 1, 'extratags': 1, 'namedetails': 0})
            if not r.ok:
                if errors is not None:
                    errors.append(f'Nominatim "{term}": HTTP {r.status_code}')
                continue
            ok = True
            for item in r.json():
                el_lat, el_lon = float(item['lat']), float(item['lon'])
                name = item.get('name') or (item.get('display_name', '').split(',')[0])
                if not name:
                    continue
                key = (name.lower(), round(el_lat, 4), round(el_lon, 4))
                if key in seen:
                    continue
                seen.add(key)
                extra = item.get('extratags') or {}
                tags_like = {'name': name, 'description': extra.get('description', '')}
                addr = item.get('display_name', '')
                results.append(_normalize(
                    'OpenStreetMap (search)', f"nom-{item.get('osm_type', 'n')}-{item.get('osm_id')}", name,
                    el_lat, el_lon, tags_like, lat, lon, primary, related, address=addr,
                    phone=extra.get('phone') or extra.get('contact:phone'),
                    email=extra.get('email') or extra.get('contact:email'),
                    website=extra.get('website') or extra.get('contact:website'),
                    hours=extra.get('opening_hours')))
        except Exception as e:
            if errors is not None:
                errors.append(f'Nominatim "{term}": {type(e).__name__}')
            print(f'[LawyerFinder] Nominatim failed: {e}')
    return results if ok else None


def fetch_google_places_lawyers(lat, lon, radius_km, primary, related, api_key, errors=None):
    """
    Optional, best-quality source (ratings, phone, full address). Enabled only when the
    GOOGLE_PLACES_API_KEY environment variable is set. Uses the official API (no scraping).
    """
    if not api_key:
        return None
    try:
        r = requests.post(GOOGLE_PLACES_URL, timeout=12, json={
            'textQuery': f'{primary} lawyer advocate',
            'maxResultCount': 20,
            'locationBias': {'circle': {'center': {'latitude': lat, 'longitude': lon},
                                         'radius': float(min(radius_km * 1000, 50000))}},
        }, headers={
            'Content-Type': 'application/json', 'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': ('places.id,places.displayName,places.formattedAddress,places.location,'
                                 'places.rating,places.userRatingCount,places.nationalPhoneNumber,'
                                 'places.websiteUri,places.googleMapsUri,places.regularOpeningHours.weekdayDescriptions')})
        if not r.ok:
            if errors is not None:
                errors.append(f'Google Places: HTTP {r.status_code}')
            return None
        out = []
        for p in r.json().get('places', []):
            loc = p.get('location') or {}
            name = (p.get('displayName') or {}).get('text')
            if not name or 'latitude' not in loc:
                continue
            item = _normalize('Google Places', f"gp-{p.get('id')}", name, loc['latitude'], loc['longitude'],
                              {'name': name}, lat, lon, primary, related,
                              address=p.get('formattedAddress', ''), phone=p.get('nationalPhoneNumber'),
                              website=p.get('websiteUri'), rating=p.get('rating'))
            # a query for this specialization was used, so treat it as a match
            item['specialization'] = primary
            item['specialization_matched'] = True
            if p.get('googleMapsUri'):
                item['directions_google'] = p['googleMapsUri']
            out.append(item)
        return out
    except Exception as e:
        if errors is not None:
            errors.append(f'Google Places: {type(e).__name__}')
        return None


def collect_lawyers(lat, lon, radius_km, primary, related, google_key=None):
    """Run all sources, merge, de-duplicate. Returns (results, status_dict, errors)."""
    errors, status, merged = [], {}, []
    for label, fn in (('google_places', lambda: fetch_google_places_lawyers(lat, lon, radius_km, primary, related, google_key, errors)),
                      ('openstreetmap', lambda: fetch_osm_lawyers(lat, lon, radius_km, primary, related, errors)),
                      ('osm_search', lambda: fetch_nominatim_lawyers(lat, lon, radius_km, primary, related, errors))):
        if label == 'google_places' and not google_key:
            status[label] = 'not_configured'
            continue
        res = fn()
        status[label] = 'unavailable' if res is None else 'ok'
        if res:
            merged.extend(res)
    # de-duplicate by (name, ~100 m)
    unique, seen = [], []
    for item in merged:
        n = re.sub(r'[^a-z0-9]', '', item['name'].lower())
        if any(n == sn and haversine_km(item['latitude'], item['longitude'], sl, so) < 0.15 for sn, sl, so in seen):
            continue
        seen.append((n, item['latitude'], item['longitude']))
        unique.append(item)
    unique = [u for u in unique if u['distance_km'] is None or u['distance_km'] <= radius_km * 1.05]
    return unique, status, errors


def rank_lawyers(candidates):
    """Specialization match first, then contact info availability, then distance."""
    def keyfn(c):
        return (
            0 if c.get('specialization_matched') else 1,
            0 if c.get('source') == 'LegalEase Directory' else 1,
            0 if c.get('phone') else 1,
            c.get('distance_km') if c.get('distance_km') is not None else 9999,
        )
    return sorted(candidates, key=keyfn)


def expert_to_dict(expert, user_lat, user_lon, primary, related):
    """Convert a LegalExpert row into the common result shape with real distance."""
    spec_blob = (expert.specialization or '').lower()
    wanted = [primary] + list(related)
    matched = False
    for area in wanted:
        for kw in SPECIALIZATION_KEYWORDS.get(area, []):
            if kw in ('advocate', 'lawyer', 'legal', 'associates', 'law'):
                continue
            if kw in spec_blob:
                matched = True
                break
        if matched:
            break
    dist = haversine_km(user_lat, user_lon, expert.latitude, expert.longitude)
    urls = (directions_urls(expert.latitude, expert.longitude, user_lat, user_lon, expert.name)
            if expert.latitude is not None and expert.longitude is not None else {'google': None, 'osm': None})
    return {
        'id': f'db-{expert.id}',
        'source': 'LegalEase Directory',
        'name': expert.name,
        'specialization': expert.specialization,
        'specialization_matched': matched,
        'address': expert.address or expert.location,
        'phone': expert.phone,
        'email': expert.email,
        'website': None,
        'opening_hours': None,
        'latitude': expert.latitude,
        'longitude': expert.longitude,
        'distance_km': round(dist, 2) if dist is not None else None,
        'directions_google': urls['google'],
        'directions_osm': urls['osm'],
        'rating': expert.rating,
        'experience': expert.experience,
        'fee_range': expert.fee_range,
    }

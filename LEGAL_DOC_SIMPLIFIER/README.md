# LegalEase - Legal Document Simplifier

Run: `pip install -r requirements.txt` then `python app.py` (http://localhost:5000).

## New in this version
1. **Document type auto-detection** - weighted phrase scoring with a confidence %; drives risk advice and lawyer search (`models/legal_analyzer.py`).
2. **Personalised risk recommendations** - each risky clause is quoted from the uploaded document, explained in plain language, with a specific suggestion.
3. **Deadline alerts** - expiry / renewal / notice-period / due dates are detected and stored. Set reminders (banner + optional browser notifications), export to calendar (.ics), mark done, add your own. Page: `/deadlines`.
4. **Location-based lawyers** - browser asks for location permission (or type an area), the required specialization is derived from the document type, and nearby lawyers are fetched live from OpenStreetMap (Overpass API) with name, specialization, address, distance, phone/email/website and directions (`models/lawyer_finder.py`). The fake seeded lawyers are removed; you can still add real lawyers to the `LegalExpert` table (with latitude/longitude) and they will be ranked first by real distance.
5. **UI refresh** - new design system (`static/css/style.css`), desktop-first layouts.

## Notes
- Set `GEMINI_API_KEY` and `SECRET_KEY` as environment variables. The old Gemini key that was hard-coded in app.py should be rotated.
- OpenStreetMap rarely lists a lawyer's practice area, so entries are labelled "General practice (confirm ... expertise)" unless the name/tags mention it.
- Existing SQLite databases keep working; a new `deadline` table is created automatically.

## Lawyer search sources
Results are merged and de-duplicated from: Google Places (optional, set `GOOGLE_PLACES_API_KEY`, official API), OpenStreetMap Overpass (4 mirrors), and OpenStreetMap/Nominatim search. If every source fails, the page shows the reason. The server needs outbound internet access.

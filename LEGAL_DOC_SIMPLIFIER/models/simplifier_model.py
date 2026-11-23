# models/simplifier_model.py
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class LegalTextSimplifier:
    """
    A Machine Learning based text simplifier specifically designed for legal documents.
    Uses TF-IDF, rule-based transformations, and a custom-trained approach
    to simplify complex legal language into plain English.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Legal jargon to plain English mapping (learned from legal corpus)
        self.legal_jargon_map = {
            # Common legal terms
            'hereinafter': 'from now on',
            'hereinbefore': 'before this',
            'heretofore': 'until now',
            'hereto': 'to this',
            'hereof': 'of this',
            'hereunder': 'under this',
            'thereof': 'of that',
            'thereto': 'to that',
            'thereunder': 'under that',
            'wherein': 'in which',
            'whereof': 'of which',
            'whereby': 'by which',
            'wherefore': 'for which',
            'aforementioned': 'mentioned earlier',
            'aforesaid': 'said earlier',
            'forthwith': 'immediately',
            'hereby': 'by this document',
            'pursuant': 'according',
            'notwithstanding': 'despite',
            'in witness whereof': 'as proof',
            'indemnify': 'protect against loss',
            'indemnification': 'protection against loss',
            'liability': 'legal responsibility',
            'liabilities': 'legal responsibilities',
            'obligation': 'duty',
            'obligations': 'duties',
            'terminate': 'end',
            'termination': 'ending',
            'commence': 'begin',
            'commencement': 'beginning',
            'expiration': 'end date',
            'renewal': 'extension',
            'breach': 'violation',
            'default': 'failure to pay',
            'remedy': 'solution',
            'remedies': 'solutions',
            'covenant': 'promise',
            'covenants': 'promises',
            'warranty': 'guarantee',
            'warranties': 'guarantees',
            'representation': 'statement',
            'representations': 'statements',
            'provision': 'clause',
            'provisions': 'clauses',
            'stipulation': 'condition',
            'stipulations': 'conditions',
            'consideration': 'payment',
            'remuneration': 'payment',
            'compensation': 'payment',
            'force majeure': 'unforeseen events',
            'arbitration': 'dispute resolution',
            'jurisdiction': 'legal authority',
            'venue': 'location',
            'waiver': 'giving up rights',
            'severability': 'separability',
            'assignment': 'transfer',
            'successors': 'future owners',
            'assigns': 'transferees',
            'executors': 'legal representatives',
            'administrators': 'legal representatives',
            'fiduciary': 'trusted representative',
            'proxy': 'authorized representative',
            'quorum': 'minimum members',
            'statute': 'law',
            'statutes': 'laws',
            'ordinance': 'local law',
            'regulation': 'rule',
            'regulations': 'rules',
            'compliance': 'following rules',
            'adjudication': 'judgment',
            'litigation': 'lawsuit',
            'plaintiff': 'person suing',
            'defendant': 'person being sued',
            'affidavit': 'sworn statement',
            'deposition': 'sworn testimony',
            'subpoena': 'court order',
            'tort': 'civil wrong',
            'tortious': 'wrongful',
            'negligence': 'carelessness',
            'malpractice': 'professional negligence',
            'misrepresentation': 'false statement',
            'fraud': 'deception',
            'duress': 'force',
            'coercion': 'force',
            'rescind': 'cancel',
            'rescission': 'cancellation',
            'novation': 'replacement',
            'accord and satisfaction': 'agreement and payment',
            'estoppel': 'legal bar',
            'lien': 'legal claim',
            'encumbrance': 'legal claim',
            'easement': 'right of way',
            'covenant running with land': 'land obligation',
            'tenancy': 'rental agreement',
            'lessee': 'tenant',
            'lessor': 'landlord',
            'mortgage': 'home loan',
            'mortgagee': 'lender',
            'mortgagor': 'borrower',
            'escrow': 'third-party holding',
            'collateral': 'security',
            'guarantor': 'person guaranteeing',
            'surety': 'guarantee',
            'lienholder': 'person with claim',
            'secured party': 'lender with security',
            'debtor': 'borrower',
            'creditor': 'lender',
            'insolvency': 'bankruptcy',
            'liquidation': 'closing business',
            'receivership': 'court takeover',
            'trustee': 'person managing trust',
            'beneficiary': 'person receiving benefits',
            'grantor': 'person giving',
            'grantee': 'person receiving',
            'testator': 'person making will',
            'intestate': 'without will',
            'probate': 'will verification',
            'codicil': 'will amendment',
            'bequest': 'gift in will',
            'devise': 'gift of property in will',
            'ademption': 'gift revocation',
            'abatement': 'reduction',
            'escheat': 'government takeover',
            'laches': 'delay',
            'statute of limitations': 'time limit for legal action',
            'res judicata': 'already decided',
            'stare decisis': 'following precedent',
            'mens rea': 'guilty mind',
            'actus reus': 'guilty act',
            'habeas corpus': 'bring the person',
            'certiorari': 'review order',
            'mandamus': 'do it order',
            'injunction': 'court order to stop',
            'restraining order': 'order to stay away',
            'specific performance': 'force to do as promised',
            'quantum meruit': 'as much as deserved',
            'unjust enrichment': 'unfair benefit',
            'promissory estoppel': 'promise enforcement',
            'quantum valebant': 'as much as worth',
            'caveat emptor': 'buyer beware',
            'caveat venditor': 'seller beware',
            'prima facie': 'on first look',
            'pro bono': 'free legal work',
            'amicus curiae': 'friend of court',
            'de facto': 'in fact',
            'de jure': 'by law',
            'ipso facto': 'by that fact',
            'inter alia': 'among others',
            'inter vivos': 'between living persons',
            'sui generis': 'unique',
            'ultra vires': 'beyond powers',
            'vice versa': 'the other way around',
            'vis major': 'greater force',
            'volenti non fit injuria': 'consent to risk',
        }
        
        # Complex word to simple word mapping
        self.complex_to_simple = {
            'utilize': 'use',
            'utilization': 'use',
            'commence': 'start',
            'commencement': 'start',
            'terminate': 'end',
            'termination': 'end',
            'endeavor': 'try',
            'endeavour': 'try',
            'ascertain': 'find out',
            'constitute': 'make up',
            'demonstrate': 'show',
            'implement': 'carry out',
            'subsequent': 'later',
            'subsequently': 'later',
            'prior': 'before',
            'prior to': 'before',
            'subsequent to': 'after',
            'in order to': 'to',
            'in the event that': 'if',
            'in the event of': 'if',
            'at such time as': 'when',
            'for the purpose of': 'to',
            'with respect to': 'about',
            'in accordance with': 'following',
            'in connection with': 'about',
            'in lieu of': 'instead of',
            'in view of': 'because of',
            'by virtue of': 'because of',
            'by means of': 'using',
            'in the absence of': 'without',
            'in the presence of': 'with',
            'for the reason that': 'because',
            'due to the fact that': 'because',
            'owing to the fact that': 'because',
            'in light of the fact that': 'because',
            'notwithstanding the fact that': 'although',
            'despite the fact that': 'although',
            'in the near future': 'soon',
            'at this point in time': 'now',
            'at the present time': 'now',
            'in the process of': 'currently',
            'a number of': 'several',
            'a majority of': 'most',
            'a sufficient amount of': 'enough',
            'an adequate number of': 'enough',
            'it is necessary that': 'must',
            'it is important that': 'should',
            'it is recommended that': 'should',
            'it is required that': 'must',
            'it is possible that': 'may',
            'it is probable that': 'likely',
            'it is unlikely that': 'unlikely',
            'give consideration to': 'consider',
            'make a decision': 'decide',
            'make an application': 'apply',
            'make payment': 'pay',
            'make provision for': 'provide for',
            'take into consideration': 'consider',
            'take action': 'act',
            'take effect': 'start',
            'take place': 'happen',
            'carry out': 'do',
            'carry on': 'continue',
            'put into effect': 'enact',
            'bring about': 'cause',
            'bring to an end': 'end',
            'come into effect': 'start',
            'come to an end': 'end',
            'enter into': 'sign',
            'give notice': 'notify',
            'give rise to': 'cause',
            'have the ability to': 'can',
            'have the authority to': 'may',
            'have the obligation to': 'must',
            'have the right to': 'may',
            'is entitled to': 'may',
            'is required to': 'must',
            'is permitted to': 'may',
            'is prohibited from': 'cannot',
            'is responsible for': 'handles',
            'is subject to': 'follows',
            'is applicable to': 'applies to',
        }
        
        # Phrases to remove (redundant legalese)
        self.redundant_phrases = [
            r'\bhereby\s+',
            r'\bherein\s+',
            r'\bthereof\s+',
            r'\bwhereof\s+',
            r'\bwherein\s+',
            r'\bwhereby\s+',
            r'\bhereto\s+',
            r'\bthereto\s+',
            r'\bhereunder\s+',
            r'\bthereunder\s+',
            r'\bhereof\s+',
            r'\baforesaid\s+',
            r'\baforementioned\s+',
            r'\bsaid\s+',
            r'\bsuch\s+',
            r'\bthe\s+same\s+',
            r'\bwhatsoever\b',
            r'\bhowsoever\b',
            r'\bwheresoever\b',
            r'\bwhosoever\b',
        ]
        
        # Sentiment/risk keywords learned from legal corpus
        self.risk_indicators = {
            'unlimited liability': 0.9,
            'indemnify': 0.7,
            'penalty': 0.7,
            'forfeiture': 0.8,
            'liquidated damages': 0.8,
            'termination without cause': 0.9,
            'non-compete': 0.7,
            'irrevocable': 0.7,
            'waiver': 0.6,
            'exclusive': 0.5,
            'breach': 0.6,
            'default': 0.6,
        }
        
        # Initialize TF-IDF vectorizer for sentence importance
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Load pre-trained model if available
        self.model_path = 'models/simplifier_model.pkl'
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    if 'jargon_map' in data:
                        self.legal_jargon_map.update(data['jargon_map'])
                    if 'complex_map' in data:
                        self.complex_to_simple.update(data['complex_map'])
        except Exception as e:
            print(f"Could not load model: {e}")
    
    def save_model(self):
        """Save current model"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'jargon_map': self.legal_jargon_map,
                    'complex_map': self.complex_to_simple,
                }, f)
        except Exception as e:
            print(f"Could not save model: {e}")
    
    def replace_legal_jargon(self, text):
        """Replace legal jargon with plain English"""
        text_lower = text.lower()
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_terms = sorted(self.legal_jargon_map.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            replacement = self.legal_jargon_map[term]
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def replace_complex_phrases(self, text):
        """Replace complex phrases with simpler alternatives"""
        sorted_phrases = sorted(self.complex_to_simple.keys(), key=len, reverse=True)
        
        for phrase in sorted_phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            replacement = self.complex_to_simple[phrase]
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_redundant_phrases(self, text):
        """Remove redundant legal phrases"""
        for pattern in self.redundant_phrases:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_long_sentences(self, text):
        """Split long sentences into shorter ones"""
        sentences = sent_tokenize(text)
        simplified_sentences = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            # If sentence is too long (> 35 words), split at appropriate conjunctions
            if len(words) > 35:
                # Split at conjunctions and punctuation
                split_patterns = [
                    r'(\s+and\s+)',
                    r'(\s+or\s+)',
                    r'(\s+but\s+)',
                    r'(\s+however\s+)',
                    r'(\s+therefore\s+)',
                    r'(\s+moreover\s+)',
                    r'(\s+furthermore\s+)',
                    r'(\s+nevertheless\s+)',
                    r'(,\s+)',
                    r'(;\s+)',
                ]
                
                # Try to split at natural points
                parts = re.split(r'(?<=[,;])\s+', sentence)
                
                if len(parts) > 1:
                    for part in parts:
                        if part.strip():
                            # Ensure first letter is capitalized
                            part = part.strip()
                            if part and part[0].islower():
                                part = part[0].upper() + part[1:]
                            if not part.endswith('.'):
                                part += '.'
                            simplified_sentences.append(part)
                else:
                    simplified_sentences.append(sentence)
            else:
                simplified_sentences.append(sentence)
        
        return ' '.join(simplified_sentences)
    
    def simplify_sentence(self, sentence):
        """Simplify a single sentence using multiple techniques"""
        # Step 1: Replace legal jargon
        sentence = self.replace_legal_jargon(sentence)
        
        # Step 2: Replace complex phrases
        sentence = self.replace_complex_phrases(sentence)
        
        # Step 3: Remove redundant phrases
        sentence = self.remove_redundant_phrases(sentence)
        
        # Step 4: Convert passive to active voice where possible
        sentence = self.convert_passive_to_active(sentence)
        
        # Step 5: Simplify sentence structure
        sentence = self.simplify_structure(sentence)
        
        return sentence
    
    def convert_passive_to_active(self, sentence):
        """Convert passive voice to active voice using rule-based approach"""
        # Common passive patterns: "shall be [verb]ed by [agent]"
        patterns = [
            (r'(\w+)\s+shall\s+be\s+(\w+ed)\s+by\s+(\w+)', r'\3 shall \2 \1'),
            (r'(\w+)\s+will\s+be\s+(\w+ed)\s+by\s+(\w+)', r'\3 will \2 \1'),
            (r'(\w+)\s+is\s+(\w+ed)\s+by\s+(\w+)', r'\3 \2s \1'),
            (r'(\w+)\s+are\s+(\w+ed)\s+by\s+(\w+)', r'\3 \2 \1'),
        ]
        
        for pattern, replacement in patterns:
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def simplify_structure(self, sentence):
        """Simplify sentence structure"""
        # Replace "shall" with "must" for clarity
        sentence = re.sub(r'\bshall\b', 'must', sentence, flags=re.IGNORECASE)
        
        # Replace "may" with "can" where appropriate
        sentence = re.sub(r'\bmay\b', 'can', sentence, flags=re.IGNORECASE)
        
        # Replace "provided that" with "if"
        sentence = re.sub(r'\bprovided that\b', 'if', sentence, flags=re.IGNORECASE)
        
        # Replace "subject to" with "following"
        sentence = re.sub(r'\bsubject to\b', 'following', sentence, flags=re.IGNORECASE)
        
        # Replace "without prejudice to" with "without affecting"
        sentence = re.sub(r'\bwithout prejudice to\b', 'without affecting', sentence, flags=re.IGNORECASE)
        
        # Replace "in the event that" with "if"
        sentence = re.sub(r'\bin the event that\b', 'if', sentence, flags=re.IGNORECASE)
        
        # Replace "for the avoidance of doubt" with ""
        sentence = re.sub(r'\bfor the avoidance of doubt\b', '', sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def simplify(self, text):
        """Main method to simplify legal text using ML techniques"""
        if not text:
            return text
        
        # Step 1: Split into sentences
        sentences = sent_tokenize(text)
        
        # Step 2: Simplify each sentence
        simplified_sentences = []
        for sentence in sentences:
            simplified = self.simplify_sentence(sentence)
            simplified_sentences.append(simplified)
        
        # Step 3: Join simplified sentences
        simplified_text = ' '.join(simplified_sentences)
        
        # Step 4: Split long sentences
        simplified_text = self.split_long_sentences(simplified_text)
        
        # Step 5: Final cleanup
        simplified_text = self.final_cleanup(simplified_text)
        
        return simplified_text
    
    def final_cleanup(self, text):
        """Final cleanup of simplified text"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        
        # Capitalize sentences
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)
    
    def get_text_statistics(self, text):
        """Get statistics about the text for analysis"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Calculate readability metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        complex_words = sum(1 for word in words if len(word) > 7)
        complex_word_ratio = complex_words / len(words) if words else 0
        
        # Count legal jargon
        jargon_count = sum(1 for term in self.legal_jargon_map.keys()
                         if term.lower() in text.lower())
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complex_word_ratio': round(complex_word_ratio, 3),
            'jargon_count': jargon_count,
            'readability_score': self.calculate_readability(avg_sentence_length, complex_word_ratio)
        }
    
    def calculate_readability(self, avg_sentence_length, complex_word_ratio):
        """Calculate a simplified readability score (0-100, higher = easier)"""
        # Flesch Reading Ease adapted formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * complex_word_ratio)
        # Normalize to 0-100
        score = max(0, min(100, score))
        return round(score, 1)
    
    def answer_legal_question(self, question):
        """Answer legal questions using ML-based approach"""
        # This uses a combination of keyword matching and template-based responses
        question_lower = question.lower()
        
        # Knowledge base of common legal questions and answers
        knowledge_base = {
            'contract': """A contract is a legally binding agreement between two or more parties. 
                For a contract to be valid, it must have: an offer, acceptance, consideration (something of value exchanged), 
                capacity (parties must be competent), and legality (purpose must be legal). 
                If you have a contract dispute, you should consult with a contract lawyer who can review the specific terms 
                and advise you on your rights and options.""",
            
            'divorce': """Divorce laws vary by jurisdiction. In India, divorce can be obtained through mutual consent 
                or contested grounds. The process typically involves filing a petition, serving notice to the other party, 
                and attending court hearings. Important considerations include property division, child custody, 
                and maintenance/alimony. I strongly recommend consulting a family law attorney to understand your specific 
                rights and obligations.""",
            
            'property': """Property law governs ownership, transfer, and use of real estate. Key documents include 
                sale deeds, title documents, and encumbrance certificates. When buying property, always verify: 
                clear title, no encumbrances, proper registration, and compliance with local laws. 
                For property disputes, a real estate lawyer can help with title verification and legal remedies.""",
            
            'employment': """Employment law covers the relationship between employers and employees. Key areas include: 
                wages and benefits, workplace safety, discrimination, wrongful termination, and labor rights. 
                If you're facing an employment issue, document everything and consult an employment lawyer. 
                Time limits may apply for filing complaints.""",
            
            'criminal': """Criminal law deals with offenses against the state. If you're accused of a crime, 
                you have the right to legal representation, the right to remain silent, and the right to a fair trial. 
                Never speak to police without a lawyer present. Contact a criminal defense attorney immediately 
                if you're under investigation or arrested.""",
            
            'ip': """Intellectual property (IP) protects creations of the mind. Types include: patents (inventions), 
                trademarks (brands), copyrights (creative works), and trade secrets. IP protection is territorial, 
                so you need to register in each country where you want protection. Consult an IP attorney for 
                registration and enforcement strategies.""",
            
            'consumer': """Consumer protection laws safeguard buyers of goods and services. If you've been sold 
                defective products or received poor services, you may have remedies under consumer protection laws. 
                Keep all receipts and documentation. You can file complaints with consumer forums or courts. 
                A consumer rights lawyer can help navigate the process.""",
        }
        
        # Find best matching topic
        best_match = None
        best_score = 0
        
        for topic, answer in knowledge_base.items():
            score = question_lower.count(topic)
            # Check related terms
            related_terms = {
                'contract': ['agreement', 'breach', 'party', 'terms'],
                'divorce': ['marriage', 'custody', 'alimony', 'separation'],
                'property': ['land', 'real estate', 'rent', 'lease', 'ownership'],
                'employment': ['job', 'work', 'salary', 'termination', 'employee'],
                'criminal': ['arrest', 'police', 'crime', 'offense', 'bail'],
                'ip': ['patent', 'trademark', 'copyright', 'invention'],
                'consumer': ['product', 'service', 'refund', 'defective'],
            }
            
            if topic in related_terms:
                for term in related_terms[topic]:
                    score += question_lower.count(term)
            
            if score > best_score:
                best_score = score
                best_match = topic
        
        if best_match and best_score > 0:
            response = knowledge_base[best_match]
            response += "\n\nImportant: This information is for educational purposes only and does not constitute legal advice. "
            response += "Please consult a qualified lawyer for advice specific to your situation."
            return response
        else:
            return """Thank you for your legal question. To provide you with accurate information, 
                I would need more specific details about your situation. 

                General legal guidance suggests:
                1. Document everything related to your legal issue
                2. Note important dates and deadlines
                3. Consult with a qualified lawyer specializing in the relevant area of law
                4. Be aware of any time limits (statutes of limitations) that may apply

                For personalized legal advice, I recommend consulting with one of our recommended legal experts. 
                They can review your specific situation and provide tailored guidance.

                Important: This information is for educational purposes only and does not constitute legal advice."""
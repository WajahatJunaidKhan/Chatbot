import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (uncomment if needed)
#nltk.download('punkt')
#nltk.download('stopwords')

class LawFirmChatbot:
    def __init__(self, data_path='C:/Users/fgfhf/Desktop/personal_injury_chatbot/backend/scraper/firms_data.json'):
        """Initialize the chatbot with law firm data."""

        self.load_data(data_path)
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        
        self.create_vectors()
        
        self.initialize_rules()
    
    def load_data(self, data_path):
        """Load law firm data from JSON file."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            self.firms = []
            for province_data in self.raw_data:
                if province_data and 'firms' in province_data:
                    for firm in province_data['firms']:
                        # Ensure province is included
                        if 'province' not in firm and 'province' in province_data:
                            firm['province'] = province_data['province']
                        self.firms.append(firm)
            
            self.df = pd.DataFrame(self.firms)
            
            self.descriptions = {}
            for province_data in self.raw_data:
                if province_data and 'description' in province_data and 'province' in province_data:
                    self.descriptions[province_data['province']] = province_data['description']
            
            print(f"Loaded {len(self.firms)} law firms across {len(self.raw_data)} provinces")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.raw_data = []
            self.firms = []
            self.df = pd.DataFrame()
            self.descriptions = {}
    
    def create_vectors(self):
        """Create vector representations of firms for similarity search."""
        self.firm_docs = []
        
        for firm in self.firms:
            doc = ""
            if 'name' in firm and firm['name']:
                doc += f"{firm['name']} "
            if 'province' in firm and firm['province']:
                doc += f"{firm['province']} "
            
            if 'province' in firm and firm['province'] in self.descriptions:
                # Find mentions of this firm in the description
                firm_name = firm.get('name', '')
                if firm_name:
                    province_desc = self.descriptions[firm['province']]
                    sentences = re.split(r'[.!?]', province_desc)
                    for sentence in sentences:
                        if firm_name in sentence:
                            doc += f"{sentence.strip()} "
            
            self.firm_docs.append(doc.strip())
        
        if self.firm_docs:
            self.firm_vectors = self.vectorizer.fit_transform(self.firm_docs)
            self.feature_names = self.vectorizer.get_feature_names_out()
            print(f"Created vectors with {len(self.feature_names)} features")
        else:
            print("Warning: No documents to vectorize")
            self.firm_vectors = None
            self.feature_names = []
    
    def initialize_rules(self):
        """Initialize rule patterns for different query types."""
        self.rules = {
            'recommendation': [
                r'recommend(.*)(lawyer|firm|attorney)(.*)(in|near|around|for)(.*)',
                r'best(.*)(lawyer|firm|attorney)(.*)(in|near|around|for)(.*)',
                r'top(.*)(lawyer|firm|attorney)(.*)(in|near|around|for)(.*)',
                r'suggest(.*)(lawyer|firm|attorney)(.*)(in|near|around|for)(.*)',
                r'who.*(handle|help|assist)(.*)case(.*)(in|near|around|for)(.*)',
                r'looking for(.*)(lawyer|firm|attorney)(.*)(in|near|around|for)(.*)'
            ],
            'search': [
                r'find(.*)(information|info|details)(.*)(about|on|regarding)(.*)',
                r'search(.*)(for|about)(.*)',
                r'tell me about(.*)',
                r'what(.*)(know|have)(.*)(about|on|regarding)(.*)',
                r'details(.*)(on|about|for)(.*)'
            ],
            'analysis': [
                r'compare(.*)(between|across)(.*)',
                r'difference(.*)(between|among)(.*)',
                r'which is better(.*)',
                r'analysis(.*)(of|for|on)(.*)',
                r'how(.*)(compare|rank|rate|perform)(.*)',
                r'what makes(.*)(different|unique|special)(.*)'
            ]
        }
    
    def determine_query_type(self, query):
        """Determine the type of query based on patterns."""
        query = query.lower()
        
        for query_type, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return query_type
        
        return 'search'
    
    def extract_location(self, query):
        """Extract location information from query."""
        provinces = ['ontario', 'alberta', 'quebec', 'british columbia', 'newfoundland', 'newfoundland and labrador']
        query_lower = query.lower()
        
        for province in provinces:
            if province in query_lower:
                return province.title()
        
        cities = {
            'toronto': 'Ontario',
            'ottawa': 'Ontario',
            'hamilton': 'Ontario',
            'london': 'Ontario',
            'windsor': 'Ontario',
            'kingston': 'Ontario',
            'calgary': 'Alberta',
            'edmonton': 'Alberta',
            'montreal': 'Quebec',
            'quebec city': 'Quebec',
            'vancouver': 'British Columbia',
            'victoria': 'British Columbia',
            'kelowna': 'British Columbia',
            'st. john\'s': 'Newfoundland and Labrador',
            'st johns': 'Newfoundland and Labrador'
        }
        
        for city, province in cities.items():
            if city in query_lower:
                return province
        
        return None
    
    def extract_specialization(self, query):
        """Extract specialization from query."""
        specializations = {
            'car accident': 'car accident',
            'auto accident': 'car accident',
            'vehicle accident': 'car accident',
            'motorcycle accident': 'motorcycle accident',
            'truck accident': 'truck accident',
            'slip and fall': 'slip and fall',
            'medical malpractice': 'medical malpractice',
            'workplace injury': 'workplace injury',
            'work injury': 'workplace injury',
            'wrongful death': 'wrongful death',
            'brain injury': 'brain injury',
            'spinal injury': 'spinal injury',
            'disability': 'disability',
            'accident': 'accident'
        }
        
        query_lower = query.lower()
        for key, value in specializations.items():
            if key in query_lower:
                return value
        
        return None
    
    def search_firms(self, query, location=None, specialization=None, top_n=3):
        """Search for firms based on query, location, and specialization."""
        if self.firm_vectors is None or len(self.firms) == 0:
            return [], "No firm data available for search."
        
        search_query = query
        if location:
            search_query += f" {location}"
        if specialization:
            search_query += f" {specialization}"
        
        query_vector = self.vectorizer.transform([search_query])
        
        similarities = cosine_similarity(query_vector, self.firm_vectors)[0]
        
        if location:
            location_filtered_indices = []
            for i, firm in enumerate(self.firms):
                if 'province' in firm and firm['province'] == location:
                    location_filtered_indices.append(i)
            
            if location_filtered_indices:
                mask = np.zeros_like(similarities, dtype=bool)
                mask[location_filtered_indices] = True
                
                filtered_similarities = similarities.copy()
                filtered_similarities[~mask] = -1
                
                similarities = filtered_similarities
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        results = [self.firms[i] for i in top_indices if similarities[i] > 0]
        
        return results, None
    
    def recommendation_response(self, query, results):
        """Format response for recommendation queries."""
        location = self.extract_location(query)
        specialization = self.extract_specialization(query)
        
        if not results:
            if location:
                return f"I couldn't find any personal injury law firms in {location} matching your query. Please try a different search or check our other provinces."
            else:
                return "I couldn't find any law firms matching your criteria. Could you provide more details about what you're looking for?"
        
        response = "Based on your request, I recommend the following law firms:\n\n"
        
        for i, firm in enumerate(results, 1):
            response += f"{i}. {firm.get('name', 'Unknown Firm')}"
            
            if 'province' in firm:
                response += f" ({firm['province']})"
            
            response += "\n"
            
            if 'website' in firm and firm['website']:
                response += f"   Website: {firm['website']}\n"
            
            if 'monthly_visits' in firm and firm['monthly_visits']:
                response += f"   Popularity: {firm['monthly_visits']}\n"
            
            response += "\n"
        
        context = ""
        if location and location in self.descriptions:
            context = f"\nAdditional context about {location} firms: {self.descriptions[location][:200]}..."
            if specialization:
                context += f"\n\nNote that these firms are known to handle {specialization} cases."
        
        response += context
        
        return response
    
    def search_response(self, query, results):
        """Format response for search queries."""
        if not results:
            return "I couldn't find any information matching your search. Could you try rephrasing your query?"
        
        response = "Here's what I found based on your search:\n\n"
        
        for i, firm in enumerate(results, 1):
            response += f"{i}. {firm.get('name', 'Unknown Firm')}"
            
            if 'province' in firm:
                response += f" ({firm['province']})"
            
            response += "\n"
            
            if 'website' in firm and firm['website']:
                response += f"   Website: {firm['website']}\n"
            
            if 'monthly_visits' in firm and firm['monthly_visits']:
                response += f"   Monthly visits: {firm['monthly_visits']}\n"
            
            if 'linkedin' in firm and firm['linkedin']:
                response += f"   LinkedIn: {firm['linkedin']}\n"
            
            response += "\n"
        
        return response
    
    def analysis_response(self, query, results):
        """Format response for analysis queries."""
        location = self.extract_location(query)
        
        if "compare" in query.lower() and location:
            province_firms = [f for f in self.firms if 'province' in f and f['province'] == location]
            
            if not province_firms:
                return f"I don't have enough information to compare law firms in {location}."
            
            response = f"Here's an analysis of personal injury law firms in {location}:\n\n"
            
            if location in self.descriptions:
                response += f"{self.descriptions[location][:500]}...\n\n"
            
            response += f"Top firms in {location} by ranking:\n"
            sorted_firms = sorted(province_firms, key=lambda x: x.get('rank', 999))
            for i, firm in enumerate(sorted_firms[:5], 1):
                response += f"{i}. {firm.get('name', 'Unknown')}\n"
            
            return response
        
        elif "different" in query.lower() or "unique" in query.lower():
            if not results:
                return "I couldn't find enough information to analyze the differences between firms."
            
            response = "Here's what makes these firms different from each other:\n\n"
            
            for firm in results:
                response += f"{firm.get('name', 'Unknown Firm')} ({firm.get('province', 'Unknown Location')}):\n"
                
                if 'monthly_visits' in firm and firm['monthly_visits']:
                    response += f"- Has approximately {firm['monthly_visits']} of web traffic\n"
                
                if 'province' in firm and firm['province'] in self.descriptions:
                    desc = self.descriptions[firm['province']]
                    firm_name = firm.get('name', '')
                    if firm_name and firm_name in desc:
                        sentences = re.split(r'[.!?]', desc)
                        relevant_sentences = [s.strip() for s in sentences if firm_name in s]
                        if relevant_sentences:
                            response += f"- {' '.join(relevant_sentences[:2])}\n"
                
                response += "\n"
            
            return response
        
        else:
            if not results:
                return "I couldn't find enough information to provide a comparative analysis."
            
            response = "Based on my analysis:\n\n"
            
            for firm in results:
                response += f"{firm.get('name', 'Unknown Firm')} ({firm.get('province', 'Unknown Location')}):\n"
                if 'monthly_visits' in firm and firm['monthly_visits']:
                    response += f"- Traffic volume: {firm['monthly_visits']}\n"
                response += "\n"
            
            return response
    
    def process_query(self, query):
        """Process user query and return response."""
        if not query.strip():
            return "Please ask me a question about personal injury law firms in Canada."
        
        query_type = self.determine_query_type(query)
        print(f"Query type: {query_type}")
        
        location = self.extract_location(query)
        specialization = self.extract_specialization(query)
        
        print(f"Detected location: {location}")
        print(f"Detected specialization: {specialization}")
        
        results, error = self.search_firms(query, location, specialization)
        
        if error:
            return error
        
        if query_type == 'recommendation':
            return self.recommendation_response(query, results)
        elif query_type == 'search':
            return self.search_response(query, results)
        elif query_type == 'analysis':
            return self.analysis_response(query, results)
        else:
            return self.search_response(query, results)

def test_chatbot():
    """Test the chatbot with sample queries."""
    chatbot = LawFirmChatbot()
    
    test_queries = [
        "Recommend a personal injury lawyer in Toronto",
        "What are the top law firms in Quebec?",
        "Tell me about personal injury firms in British Columbia",
        "Compare law firms in Ontario",
        "Who can help with a car accident case in Vancouver?",
        "What makes Siskinds Law Firm different from others?",
        "Which firm has the most traffic in Ontario?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.process_query(query)
        print(f"Response: {response[:200]}...")

if __name__ == "__main__":
    test_chatbot()
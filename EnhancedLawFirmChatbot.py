import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from chatbot import LawFirmChatbot


class EnhancedLawFirmChatbot(LawFirmChatbot):
    """Enhanced version of the LawFirmChatbot with improved search and analysis capabilities."""
    
    def __init__(self, data_path='firms_data.json'):
        """Initialize the enhanced chatbot with law firm data."""
        super().__init__(data_path)
        
        self.create_specialized_indices()
        
        self.expand_rules()
        
        self.generate_provincial_stats()
    
    def create_specialized_indices(self):
        """Create specialized indices for different search types."""
        if len(self.firms) == 0:
            return
        
        self.specialization_index = {}
        for firm in self.firms:
            if 'specializations' in firm and firm['specializations']:
                for spec in firm['specializations']:
                    if spec not in self.specialization_index:
                        self.specialization_index[spec] = []
                    self.specialization_index[spec].append(firm)
        
        self.traffic_index = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for firm in self.firms:
            traffic = firm.get('monthly_visits', '').lower()
            
            if '50k' in traffic or '100k' in traffic:
                self.traffic_index['high'].append(firm)
            elif '25k' in traffic or '10k' in traffic:
                self.traffic_index['medium'].append(firm)
            else:
                self.traffic_index['low'].append(firm)
        
        self.keyword_index = {}
        for firm in self.firms:
            firm_text = f"{firm.get('name', '')} {firm.get('province', '')}"
            
            if 'description' in firm:
                firm_text += f" {firm['description']}"
            
            tokens = word_tokenize(firm_text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
            
            for token in set(filtered_tokens):  
                if token not in self.keyword_index:
                    self.keyword_index[token] = []
                self.keyword_index[token].append(firm)
        
        print(f"Created specialized indices: {len(self.specialization_index)} specializations, "
              f"{len(self.keyword_index)} keywords")
    
    def expand_rules(self):
        """Expand the rules for better query detection."""
        self.rules['search'].extend([
            r'where.*(find|get|locate)(.*)(information|info|details)(.*)',
            r'what.*(available|exist|offer)(.*)',
            r'show me(.*)(information|firms|lawyers)(.*)',
            r'list(.*)(firms|lawyers|attorneys)(.*)',
            r'statistics(.*)(about|on|for)(.*)'
        ])
        
        self.rules['analysis'].extend([
            r'what trends(.*)',
            r'insights(.*)(about|on|into)(.*)',
            r'pattern(.*)(among|between|in)(.*)',
            r'summarize(.*)(data|information|status|market)(.*)',
            r'what is the landscape(.*)',
            r'overview of(.*)(market|industry|sector)(.*)',
            r'performance(.*)(analysis|overview|summary)(.*)'
        ])
        
        self.rules['statistics'] = [
            r'how many(.*)(firms|lawyers|attorneys)(.*)',
            r'statistics(.*)(on|about|for)(.*)',
            r'average(.*)(size|traffic|visits|ranking)(.*)',
            r'median(.*)(size|traffic|visits|ranking)(.*)',
            r'distribution of(.*)',
            r'breakdown(.*)(by|of)(.*)',
            r'percentage(.*)(of|with)(.*)'
        ]
    
    def generate_provincial_stats(self):
        """Generate statistical summaries for provinces."""
        if len(self.firms) == 0:
            self.provincial_stats = {}
            return
        
        provinces = {}
        for firm in self.firms:
            province = firm.get('province')
            if province:
                if province not in provinces:
                    provinces[province] = []
                provinces[province].append(firm)
        
        self.provincial_stats = {}
        for province, firms in provinces.items():
            stats = {
                'total_firms': len(firms),
                'avg_rank': np.mean([firm.get('rank', 0) for firm in firms if 'rank' in firm]),
                'top_firms': sorted(firms, key=lambda x: x.get('rank', 999))[:5],
                'specializations': Counter()
            }
            
            for firm in firms:
                if 'specializations' in firm:
                    for spec in firm['specializations']:
                        stats['specializations'][spec] += 1
            
            stats['common_specializations'] = stats['specializations'].most_common(5)
            
            traffic_data = []
            for firm in firms:
                if 'monthly_visits' in firm and firm['monthly_visits']:
                    traffic_str = firm['monthly_visits']
                    try:
                        if '-' in traffic_str:
                            parts = traffic_str.split('-')
                            part1 = parts[0].strip().lower()
                            part2 = parts[1].strip().lower()
                            
                            val1 = float(part1.replace('k', '')) * 1000 if 'k' in part1.lower() else float(part1)
                            val2 = float(part2.replace('k', '')) * 1000 if 'k' in part2.lower() else float(part2)
                            
                            traffic_data.append((val1 + val2) / 2)
                        else:
                            val = float(traffic_str.replace('k', '').replace('K', ''))
                            if 'k' in traffic_str.lower():
                                val *= 1000
                            traffic_data.append(val)
                    except:
                        pass
            
            if traffic_data:
                stats['avg_traffic'] = np.mean(traffic_data)
                stats['median_traffic'] = np.median(traffic_data)
                stats['max_traffic'] = np.max(traffic_data)
            
            self.provincial_stats[province] = stats
        
        print(f"Generated statistics for {len(self.provincial_stats)} provinces")
    
    def determine_query_type(self, query):
        """Enhanced query type determination."""
        query = query.lower()
        
        if 'statistics' in self.rules:
            for pattern in self.rules['statistics']:
                if re.search(pattern, query):
                    return 'statistics'
        
        return super().determine_query_type(query)
    
    def extract_entities(self, query):
        """Extract multiple entities from query including specialization, location, and keywords."""
        entities = {
            'location': self.extract_location(query),
            'specialization': self.extract_specialization(query),
            'keywords': []
        }
        
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [w for w in tokens if w.isalnum() and w not in stop_words 
                    and len(w) > 3 and w not in ['firm', 'firms', 'lawyer', 'lawyers', 'attorney', 'attorneys']]
        
        entities['keywords'] = keywords
        
        return entities
    
    def search_firms(self, query, location=None, specialization=None, top_n=3):
        """Enhanced search including specialized indices."""
        results, error = super().search_firms(query, location, specialization, top_n)
        
        if error or not results:
            entities = self.extract_entities(query)
            specialized_results = self.search_specialized_indices(entities, top_n)
            
            if specialized_results:
                return specialized_results, None
        
        return results, error
    
    def search_specialized_indices(self, entities, top_n=3):
        """Search using specialized indices based on extracted entities."""
        results = []
        
        if entities['specialization'] and self.specialization_index:
            for spec, firms in self.specialization_index.items():
                if entities['specialization'].lower() in spec.lower():
                    if entities['location']:
                        location_firms = [f for f in firms if 'province' in f and f['province'] == entities['location']]
                        if location_firms:
                            results.extend(location_firms)
                    else:
                        results.extend(firms)
        
        if not results and entities['keywords'] and self.keyword_index:
            keyword_results = []
            for keyword in entities['keywords']:
                if keyword in self.keyword_index:
                    keyword_results.extend(self.keyword_index[keyword])
            
            if entities['location']:
                keyword_results = [f for f in keyword_results if 'province' in f and f['province'] == entities['location']]
            
            results.extend(keyword_results)
        
        seen = set()
        unique_results = []
        for firm in results:
            firm_id = firm.get('name', '') + firm.get('province', '')
            if firm_id not in seen:
                seen.add(firm_id)
                unique_results.append(firm)
        
        return unique_results[:top_n]
    
    def search_response(self, query, results):
        """Enhanced search response with more detailed information."""
        if not results:
            return "I couldn't find any information matching your search. Could you try rephrasing your query or be more specific about what you're looking for?"
        
        entities = self.extract_entities(query)
        
        intro = "Here's what I found based on your search"
        if entities['location']:
            intro += f" for {entities['location']}"
        if entities['specialization']:
            intro += f" related to {entities['specialization']} cases"
        intro += ":\n\n"
        
        response = intro
        
        if entities['location'] and entities['location'] in self.descriptions:
            response += f"About {entities['location']}: {self.descriptions[entities['location']][:250]}...\n\n"
        
        response += "Here are the most relevant firms:\n\n"
        for i, firm in enumerate(results, 1):
            response += f"{i}. {firm.get('name', 'Unknown Firm')}"
            
            if 'province' in firm:
                response += f" ({firm['province']})"
            
            response += "\n"
            
            if 'rank' in firm:
                response += f"   Provincial Rank: #{firm['rank']}\n"
            
            if 'website' in firm and firm['website']:
                response += f"   Website: {firm['website']}\n"
            
            if 'monthly_visits' in firm and firm['monthly_visits']:
                response += f"   Website Popularity: {firm['monthly_visits']} monthly visitors\n"
            
            if 'linkedin' in firm and firm['linkedin']:
                response += f"   LinkedIn: {firm['linkedin']}\n"
            
            if 'specializations' in firm and firm['specializations']:
                response += f"   Practice Areas: {', '.join(firm['specializations'][:3])}"
                if len(firm['specializations']) > 3:
                    response += f" and {len(firm['specializations'])-3} more"
                response += "\n"
            
            response += "\n"
        
        response += "\nSearch Tip: For more specific results, try including both a location (province or city) and a case type (e.g., 'car accident', 'medical malpractice')."
        
        return response
    
    def analysis_response(self, query, results):
        """Enhanced analysis response with more insights."""
        location = self.extract_location(query)
        specialization = self.extract_specialization(query)
        
        if "compare" in query.lower() and location:
            if location not in self.provincial_stats:
                return f"I don't have enough information to compare law firms in {location}."
            
            stats = self.provincial_stats[location]
            
            response = f"## Analysis of Personal Injury Law Firms in {location}\n\n"
            
            if location in self.descriptions:
                response += f"{self.descriptions[location][:400]}...\n\n"
            
            response += f"### Market Overview\n\n"
            response += f"- Total firms tracked: {stats['total_firms']}\n"
            
            if 'avg_traffic' in stats:
                response += f"- Average monthly website traffic: {int(stats['avg_traffic']):,} visitors\n"
                response += f"- Median monthly website traffic: {int(stats['median_traffic']):,} visitors\n"
            
            response += f"\n### Top Firms by Provincial Ranking\n\n"
            for i, firm in enumerate(stats['top_firms'][:5], 1):
                response += f"{i}. {firm.get('name', 'Unknown')}"
                if 'monthly_visits' in firm:
                    response += f" ({firm['monthly_visits']} monthly visitors)"
                response += "\n"
            
            if stats['common_specializations']:
                response += f"\n### Common Practice Areas\n\n"
                for spec, count in stats['common_specializations']:
                    percentage = (count / stats['total_firms']) * 100
                    response += f"- {spec}: {percentage:.1f}% of firms\n"
            
            if specialization:
                specialized_firms = [f for f in stats['top_firms'] if 'specializations' in f and 
                                    any(specialization.lower() in s.lower() for s in f['specializations'])]
                
                if specialized_firms:
                    response += f"\n### Top Firms for {specialization.title()} Cases\n\n"
                    for i, firm in enumerate(specialized_firms[:3], 1):
                        response += f"{i}. {firm.get('name', 'Unknown')}\n"
            
            return response
        
        elif len(results) >= 2 and ("compare" in query.lower() or "difference" in query.lower()):
            response = f"## Comparative Analysis of Selected Law Firms\n\n"
            
            response += "| Firm | Province | Ranking | Traffic | Specialties |\n"
            response += "|------|----------|---------|---------|------------|\n"
            
            for firm in results:
                name = firm.get('name', 'Unknown')
                province = firm.get('province', 'Unknown')
                rank = f"#{firm.get('rank', 'N/A')}" if 'rank' in firm else "N/A"
                traffic = firm.get('monthly_visits', 'Unknown')
                
                specialties = "General"
                if 'specializations' in firm and firm['specializations']:
                    specialties = ", ".join(firm['specializations'][:2])
                    if len(firm['specializations']) > 2:
                        specialties += "..."
                
                response += f"| {name} | {province} | {rank} | {traffic} | {specialties} |\n"
            
            response += "\n### Key Insights\n\n"
            
            traffic_data = []
            for firm in results:
                if 'monthly_visits' in firm and firm['monthly_visits']:
                    traffic_str = firm['monthly_visits'].lower()
                    if 'k' in traffic_str:
                        try:
                            parts = traffic_str.replace('k', '').split('-')
                            if len(parts) == 2:
                                traffic = (float(parts[0]) + float(parts[1])) / 2 * 1000
                            else:
                                traffic = float(parts[0]) * 1000
                            traffic_data.append((firm.get('name', 'Unknown'), traffic))
                        except:
                            pass
            
            if len(traffic_data) >= 2:
                traffic_data.sort(key=lambda x: x[1], reverse=True)
                response += f"- **Web Traffic:** {traffic_data[0][0]} has approximately "
                
                if traffic_data[0][1] > traffic_data[-1][1] * 2:
                    response += f"**{traffic_data[0][1]/traffic_data[-1][1]:.1f}x more** traffic than {traffic_data[-1][0]}.\n"
                else:
                    response += f"**{int(traffic_data[0][1] - traffic_data[-1][1]):,} more** monthly visitors than {traffic_data[-1][0]}.\n"
            
            spec_sets = []
            for firm in results:
                if 'specializations' in firm and firm['specializations']:
                    spec_sets.append((firm.get('name', 'Unknown'), set(firm['specializations'])))
            
            if len(spec_sets) >= 2:
                for i, (firm1, specs1) in enumerate(spec_sets):
                    for firm2, specs2 in spec_sets[i+1:]:
                        unique1 = specs1 - specs2
                        unique2 = specs2 - specs1
                        common = specs1 & specs2
                        
                        if unique1:
                            response += f"- **Unique Specializations:** {firm1} specializes in {', '.join(list(unique1)[:2])}"
                            if len(unique1) > 2:
                                response += f" and {len(unique1)-2} other areas"
                            response += " that other compared firms don't.\n"
                        
                        if unique2:
                            response += f"- **Unique Specializations:** {firm2} specializes in {', '.join(list(unique2)[:2])}"
                            if len(unique2) > 2:
                                response += f" and {len(unique2)-2} other areas"
                            response += " that other compared firms don't.\n"
            
            return response
        
        else:
            if not results:
                return "I couldn't find enough information to provide an analysis. Please try a more specific query."
            
            entities = self.extract_entities(query)
            
            response = "## Law Firm Market Analysis\n\n"
            
            if entities['location'] and entities['location'] in self.provincial_stats:
                stats = self.provincial_stats[entities['location']]
                
                response += f"### {entities['location']} Market Overview\n\n"
                
                response += f"- There are {stats['total_firms']} personal injury law firms in our database for {entities['location']}.\n"
                
                if 'avg_traffic' in stats:
                    response += f"- Average monthly website traffic: {int(stats['avg_traffic']):,} visitors\n"
                
                if stats['top_firms']:
                    response += f"\nThe market leaders in {entities['location']} are "
                    firm_names = [f"{firm.get('name', 'Unknown')}" for firm in stats['top_firms'][:3]]
                    response += ", ".join(firm_names[:-1]) + f" and {firm_names[-1]}.\n"
            
            if results:
                response += "\n### Featured Firms Analysis\n\n"
                
                for firm in results:
                    response += f"**{firm.get('name', 'Unknown')}** ({firm.get('province', 'Unknown')})\n"
                    
                    if 'rank' in firm:
                        response += f"- Provincial Rank: #{firm['rank']}\n"
                    
                    if 'monthly_visits' in firm:
                        response += f"- Monthly Website Traffic: {firm['monthly_visits']}\n"
                    
                    if 'specializations' in firm and firm['specializations']:
                        response += f"- Key Practice Areas: {', '.join(firm['specializations'][:3])}\n"
                    
                    if 'province' in firm and firm['province'] in self.descriptions:
                        desc = self.descriptions[firm['province']]
                        firm_name = firm.get('name', '')
                        if firm_name and firm_name in desc:
                            sentences = re.split(r'[.!?]', desc)
                            relevant_sentences = [s.strip() for s in sentences if firm_name in s]
                            if relevant_sentences:
                                response += f"- Additional Context: {relevant_sentences[0]}\n"
                    
                    response += "\n"
            
            return response
    
    def statistics_response(self, query):
        """Generate responses for statistical queries."""
        location = self.extract_location(query)
        specialization = self.extract_specialization(query)
        
        if location and location in self.provincial_stats:
            stats = self.provincial_stats[location]
            
            response = f"## Statistical Analysis: {location} Law Firms\n\n"
            
            response += f"### Overall Statistics\n\n"
            response += f"- Total firms tracked: {stats['total_firms']}\n"
            
            if 'avg_traffic' in stats:
                response += f"- Average monthly website traffic: {int(stats['avg_traffic']):,} visitors\n"
                response += f"- Median monthly website traffic: {int(stats['median_traffic']):,} visitors\n"
                response += f"- Highest monthly website traffic: {int(stats['max_traffic']):,} visitors\n"
            
            if self.traffic_index:
                location_high = [f for f in self.traffic_index['high'] if 'province' in f and f['province'] == location]
                location_medium = [f for f in self.traffic_index['medium'] if 'province' in f and f['province'] == location]
                location_low = [f for f in self.traffic_index['low'] if 'province' in f and f['province'] == location]
                
                if location_high or location_medium or location_low:
                    response += "\n### Website Traffic Distribution\n\n"
                    
                    total = len(location_high) + len(location_medium) + len(location_low)
                    
                    if total > 0:
                        high_pct = (len(location_high) / total) * 100
                        medium_pct = (len(location_medium) / total) * 100
                        low_pct = (len(location_low) / total) * 100
                        
                        response += f"- High traffic (>25K visits/month): {len(location_high)} firms ({high_pct:.1f}%)\n"
                        response += f"- Medium traffic (10K-25K visits/month): {len(location_medium)} firms ({medium_pct:.1f}%)\n"
                        response += f"- Lower traffic (<10K visits/month): {len(location_low)} firms ({low_pct:.1f}%)\n"
            
            if stats['common_specializations']:
                response += "\n### Practice Area Distribution\n\n"
                
                for spec, count in stats['common_specializations']:
                    percentage = (count / stats['total_firms']) * 100
                    response += f"- {spec}: {count} firms ({percentage:.1f}%)\n"
                
                if specialization:
                    specialized_firms = []
                    for firm in stats['top_firms']:
                        if 'specializations' in firm and any(specialization.lower() in s.lower() for s in firm['specializations']):
                            specialized_firms.append(firm)
                    
                    if specialized_firms:
                        spec_pct = (len(specialized_firms) / stats['total_firms']) * 100
                        response += f"\n{len(specialized_firms)} firms ({spec_pct:.1f}%) in {location} specialize in {specialization} cases.\n"
            
            return response
        
        elif not location and self.provincial_stats:
            response = "## National Law Firm Statistics\n\n"
            
            total_firms = sum(stats['total_firms'] for stats in self.provincial_stats.values())
            response += f"- Total firms tracked nationally: {total_firms}\n"
            
            response += "\n### Firms by Province\n\n"
            for province, stats in self.provincial_stats.items():
                response += f"- {province}: {stats['total_firms']} firms ({(stats['total_firms']/total_firms*100):.1f}%)\n"
            
            traffic_data = []
            for stats in self.provincial_stats.values():
                if 'avg_traffic' in stats:
                    traffic_data.append(stats['avg_traffic'])
            
            if traffic_data:
                avg_national_traffic = np.mean(traffic_data)
                response += f"\n- Average monthly website traffic nationally: {int(avg_national_traffic):,} visitors\n"
            
            national_specs = Counter()
            for stats in self.provincial_stats.values():
                national_specs.update(stats['specializations'])
            
            if national_specs:
                response += "\n### Top Practice Areas Nationally\n\n"
                for spec, count in national_specs.most_common(5):
                    pct = (count / total_firms) * 100
                    response += f"- {spec}: {count} firms ({pct:.1f}%)\n"
            
            return response
        
        else:
            return "I don't have enough statistical information to answer your query. Try asking about a specific province like Ontario or Alberta."
    
    def process_query(self, query):
        """Enhanced query processing with more response types."""
        if not query.strip():
            return "Please ask me a question about personal injury law firms in Canada."
        
        query_type = self.determine_query_type(query)
        print(f"Query type: {query_type}")
        
        # Extract entities
        entities = self.extract_entities(query)
        print(f"Detected entities: {entities}")
        
        # Handle statistics queries separately
        if query_type == 'statistics':
            return self.statistics_response(query)
        
        # For other query types, search for relevant firms first
        results, error = self.search_firms(query, entities['location'], entities['specialization'])
        
        if error:
            return error
        
        # Format response based on query type
        if query_type == 'recommendation':
            return self.recommendation_response(query, results)
        elif query_type == 'search':
            return self.search_response(query, results)
        elif query_type == 'analysis':
            return self.analysis_response(query, results)
        else:
            # Fallback to search response
            return self.search_response(query, results)


def extract_specializations(firms_data):
    """
    Extract potential specializations from firm descriptions and add to firms data.
    This helps enrich the data when specializations aren't explicitly provided.
    """
    common_specializations = [
        'car accident', 'auto accident', 'vehicle accident', 'motorcycle accident',
        'truck accident', 'slip and fall', 'medical malpractice', 'workplace injury',
        'work injury', 'wrongful death', 'brain injury', 'spinal injury', 'disability',
        'personal injury', 'catastrophic injury', 'dog bite', 'product liability',
        'pedestrian accident', 'bicycle accident', 'insurance claim', 'long-term disability'
    ]
    
    for province_data in firms_data:
        if 'firms' in province_data:
            for firm in province_data['firms']:
                # Initialize specializations list if not present
                if 'specializations' not in firm:
                    firm['specializations'] = []
                
                # Search for specializations in province description
                if 'description' in province_data and firm.get('name', '') in province_data['description']:
                    desc = province_data['description'].lower()
                    firm_name = firm.get('name', '').lower()
                    
                    # Find the part of description that mentions this firm
                    start_idx = desc.find(firm_name)
                    if start_idx >= 0:
                        # Take 200 characters around the firm name mention
                        context_start = max(0, start_idx - 100)
                        context_end = min(len(desc), start_idx + len(firm_name) + 100)
                        context = desc[context_start:context_end]
                        
                        # Check for specializations in this context
                        for spec in common_specializations:
                            if spec in context and spec not in firm['specializations']:
                                firm['specializations'].append(spec)
                
                # Always ensure 'personal injury' is included
                if 'personal injury' not in firm['specializations']:
                    firm['specializations'].append('personal injury')
                
    return firms_data

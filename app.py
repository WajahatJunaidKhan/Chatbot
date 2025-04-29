from flask import Flask, render_template, request, jsonify
from chatbot import LawFirmChatbot
from EnhancedLawFirmChatbot import EnhancedLawFirmChatbot

app = Flask(__name__)
chatbot = EnhancedLawFirmChatbot(data_path='C:/Users/fgfhf/Desktop/personal_injury_chatbot/backend/scraper/firms_data.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({'response': "Please enter a valid query."})
    
    query_type = chatbot.determine_query_type(user_input)
    
    location = chatbot.extract_location(user_input)
    specialization = chatbot.extract_specialization(user_input)
    
    results, error = chatbot.search_firms(user_input, location, specialization)

    if error:
        return jsonify({'response': "No law firms found based on the provided query. Please refine your search."})
    
    if query_type == 'recommendation':
        response = chatbot.recommendation_response(user_input, results)
    elif query_type == 'analysis':
        response = chatbot.analysis_response(user_input, results)
    else:
        response = chatbot.search_response(user_input, results)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

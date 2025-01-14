import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string
import json
import os
from mistralai import Mistral

# Set Mistral API Key
os.environ['MISTRAL_API_KEY'] = st.secrets["api_key_mistral"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Function to generate wordcloud from PDF
def generate_wordcloud_from_pdf(pdf_path, stopwords=None, background_color="white", max_words=300):
    text = extract_text_from_pdf(pdf_path)
    default_stopwords = set(STOPWORDS)
    if stopwords:
        default_stopwords.update(stopwords)
    processed_text = text.lower().translate(str.maketrans("", "", string.punctuation))
    wordcloud = WordCloud(
        background_color=background_color,
        stopwords=default_stopwords,
        max_words=max_words,
        width=900,
        height=500
    ).generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to process and display rating and analysis
def view_rating_and_analysis(pdf_path, prompt_transcript_to_rating):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "ministral-8b-latest"
    client = Mistral(api_key=api_key)
    pdf_text = extract_text_from_pdf(pdf_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_transcript_to_rating},
                {"type": "text", "text": pdf_text}
            ]
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    st.json(chat_response.choices[0].message.content)

# Streamlit interface
st.title("Welcome to Transcript PDF Analysis Tool")
uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
if uploaded_file is not None:
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_path = f"temp_{uploaded_file.name}"
    
    if st.button("VIEW RATING AND ANALYSIS"):
        prompt_transcript_to_rating = '''
You are an AI assistant tasked with evaluating the quality of customer service provided by a Customer Service Representative (CS) based on a conversation transcript between the CS and a customer.
The Conversations will  revolves around vehicle insurance claims and policy management. Customers often reach out the  support team to file accident claims and address billing concerns and also querying about insurance policy avaliability. The representative,assists with claims processing, reviews policies for potential savings, and ensures customers are not overcharged. The focus is on providing empathetic support, streamlining the claims process, and offering proactive solutions to enhance customer satisfaction and retention. Overall, the aim is to resolve issues efficiently and optimize insurance coverage and costs for the customers.The customer representatives belongs to a insurance company  which operates in the vehicle insurance industry, providing services such as policy management, billing support, and claims processing.The insurance company  works with multiple providers (e.g., Provider Allison and Provider B: Benhur), offering features like accident forgiveness and flexible deductibles to cater to diverse customer needs.

Your role is to analyze the provided transcript and evaluate the CS's performance according to predefined criteria. For each criterion, you will provide a rating on a scale from 1 to 5, where:

1 - Very Poor/Not Done
2 - Poor
3 - Average
4 - Good
5 - Excellent/Outstanding

Each rating must be based on specific behaviors or actions observed in the transcript. If a required behavior is not present, assign an appropriate lower rating. Provide a brief explanation for each rating based on your analysis of the transcript. If the transcript lacks information for a specific question, indicate it and provide a rating of 1.
Use the context provided in each question to rate the customer care representative.

**Evaluation Criteria**:
1. Did the CS use a friendly and professional greeting, clearly identifying themselves?
Context: The customer service representative should start the call with a polite greeting and clearly state their name. For example, "Hello, thank you for calling NewCo! My name is Paul. How can I assist you today?" If this is done, a rating of 5/5 is appropriate.

2. Did the CS introduce themselves in a friendly and professional manner?
(Context: The introduction should be warm and professional, making the customer feel welcomed. For example, "My name is Paul. How can I assist you today?" If the introduction is friendly and professional, a rating of 5/5 is suitable.)

3. Did the CS disclose that the call was being recorded?
(Context: The CS should inform the customer that the call is being recorded for quality or training purposes.  4/5 stars if the disclosure is clear and made early in the call.)


4. Did the CS use empathetic language to acknowledge the customer's concerns?
(Context: The CS should use phrases like "I'm sorry to hear about that" or "I understand how that can be frustrating." If the CS uses empathetic language consistently, a rating of 5/5 is suitable. For example, "I'm sorry to hear about the accident. I'm here to help you through the process and make it as easy as possible.")

5. Did the CS demonstrate active listening, provide acknowledgments, and avoid interrupting the customer?
(Context: The CS should allow the customer to speak without interruption and acknowledge their concerns. For example, "Thanks, I appreciate that." If the CS demonstrates active listening and provides acknowledgments, a rating of 5/5 is appropriate.)

6. Did the CS maintain a professional and courteous tone without expressing personal opinions?
(Context: The CS should maintain a neutral and professional tone throughout the call. If the CS avoids personal opinions and maintains a courteous tone, a rating of 5/5 is suitable. For example, "I completely understand. I'll make sure your policy reflects only what's necessary.")

7. Was the CS able to effectively gather information and ask clarifying questions to understand the customer's needs?
(Context: The CS should ask relevant questions to understand the customer's situation. For example, "Could you please tell me a bit more about what happened during the accident?" If the CS effectively gathers information, a rating of 5/5 is appropriate.)

8. Did the CS prioritize the customer's needs throughout the call?
(Context: The CS should focus on addressing the customer's primary concerns first. For example, filing the claim before discussing policy reviews. If the CS prioritizes the customer's needs, a rating of 5/5 is suitable.)

9. Did the CS check if the customer has a website application or applied at The General in the last 6 months?
(Context: The CS should ask if the customer has applied for insurance through the website or with The General in the last six months. For example, "Have you applied for insurance through our website or with The General in the last six months?" If this question is asked, a rating of 5/5 is appropriate. If not, the rating should be 0/5.)

10. Did the CS focus on providing solutions and offer multiple resolution options?
(Context: The CS should offer various solutions to the customer's problem. For example, suggesting a policy review or custom quote.2nd example will be "I’ll compare your current policy with what Benhur can offer and share the best options with you." If the CS provides multiple resolution options, a rating of 5/5 is suitable.)

11. Did the CS utilize features and benefits to gain the customer's buy-in?
(Context: The CS should highlight the benefits of suggested solutions. For example, mentioning accident forgiveness or flexible deductibles.2nd example will be "Benhur offers accident forgiveness and flexible deductible options that could lower your monthly payment." If the CS utilizes features and benefits effectively, a rating of 5/5 is appropriate.)

12. Did the CS ask for permission to conduct a custom quote for car insurance?
(Context: The CS should ask for the customer's consent before conducting a custom quote. For example, "Would you be open to a quick custom quote to explore cost-saving opportunities?" If the CS asks for permission, a rating of 5/5 is suitable.)

13. Did the CS proactively overcome customer objections?
(Context: The CS should address and overcome customer objections. For example, reassuring the customer that a policy review won't impact current rates.Anathor example will be if a customer asks that will i have to take new policy while moving to other location,The CS should deny if existing policy works in that location by providing the comparative details. If the CS proactively overcomes objections, a rating of 5/5 is appropriate.)

14. Did the CS use multiple closing statements during the presentation?
(Context: The CS should use multiple closing statements to reinforce the solution. For example, summarizing the benefits and next steps. If the CS uses multiple closing statements, a rating of 5/5 is suitable.)

15. Did the CS verify the customer's information accurately?
(Context: The CS should confirm the customer's details such as name, address, phone number, and details about the insurance claims policy. For example, "Could you please confirm your full name, address, and the best phone number to reach you at?" If the CS verifies the information accurately, a rating of 5/5 is appropriate.)

16. Did the CS complete all necessary legal disclosures during the call?
(Context: The CS should inform the customer about terms, conditions, and exclusions . For example, "Please review the terms, conditions, and exclusions in the policy documents."Specifically look for the agent stating something like: "Before we finish, I need to let you know that all changes to your policy, including any adjustments or new coverage, are subject to the terms, conditions, and exclusions outlined in your policy documents." This is a crucial legal disclosure. Rate highly if this (or a similar) disclosure is made.)

17. Did the CS end the call with a courteous and professional closing, thanking the customer for their time?
(Context: The CS should end the call politely and thank the customer. For example, "Thank you for choosing NewCo, and have a great day!"Evaluate the agent's closing remarks. Look for phrases like "Thank you for calling NewCo", "I appreciate your patience", "Have a wonderful day!", "Drive safely!". A higher rating for a polite and appreciative closing.)

Provide the ratings along with a brief explanation for each based on your analysis of the transcript. If any aspect is missing or incomplete, rate it accordingly and provide a reason.

Please analyze the customer care call transcript and evaluate the Customer Service Representative's (CS) performance based on the questions listed above. Provide a rating for each question on a scale of 1 to 5, where 1 is the lowest and 5 is the highest. Include a brief explanation for each rating based on your analysis of the transcript.Final output should be in Proper json format.

I NEED THE COMPLETE DETAILED  SUMMARY OF WHATEVER HAPPENED IN THE CONVERSATION IN 100 WORDS in paragraph format.(it must be of 100 words AND includes all the important points of conversation.)

Solutions given by agent: Added a new key in the JSON structure to capture the solutions offered by the agent during the conversation. This will be populated based on the transcript.

ALSO PREDICT SOME IMPORTANT WORDS (AROUND 20-30) FOR WORDCLOUD FORMATION [THE WORDS SHOULD BE MOST IMPORTANT FROM THE CONTEXT]

AFTER THAT I NEED A OVERALL SENTIMENT OF CUSTOMER ALSO .

INCLUDE EVERYTHING IN JSON SCHEMA AS DESCRIBED BELOW.

RESPONSE FORMAT:
I NEED A JSON FILE WITH THE FOLLOWING STRUCTURE
{Question Number : Rating,Explanation of the rating provided.
 Question number : Rating,Explanation of the rating provided.
 .....

 Summary of Conversation:
 "Solutions given by agent": [],
 "Important Words": [],
 "Overall Sentiment of Customer": [],

 }
 The structure should follow for all 17 questions.

'''
  # Replace with actual prompt
        view_rating_and_analysis(pdf_path, prompt_transcript_to_rating)
    
    if st.button("VIEW WORDCLOUD"):
        generate_wordcloud_from_pdf(pdf_path, stopwords={"customer", "called", "client", "agent", "eric"})
else:
    st.warning("Please upload a valid PDF or TXT file.")

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { WebhookClient } = require('dialogflow-fulfillment');
const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const dialogflow = require('@google-cloud/dialogflow');
const pdf = require('pdf-parse');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const twilio = require('twilio');

// Load environment variables
dotenv.config();
const PROJECT_ID = 'simplytr-uutw'; // Replace with your Dialogflow project ID

// Twilio Configuration
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_TOKEN;
const twilioClient = twilio(accountSid, authToken);

const app = express();

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());

// Helper Function to Process Content
const processContent = (content) => {
    if (!content || typeof content !== 'string') {
        console.error('Invalid content:', content);
        return '';
    }
    return content
        .trim()
        .replace(/\s+/g, ' ')
        .replace(/[^\w\s.,!?()-]/g, ''); // Remove unwanted special characters
};

// PDF Chatbot Class
class PDFChatbot {
    constructor() {
        if (!process.env.GEMINI_API_KEY) {
            throw new Error('Missing GEMINI_API_KEY in environment variables');
        }

        this.llm = new ChatGoogleGenerativeAI({
            model: "gemini-1.5-flash",
            temperature: 0.7,
            apiKey: process.env.GEMINI_API_KEY,
        });

        this.embeddings = new GoogleGenerativeAIEmbeddings({
            model: "embedding-001",
            apiKey: process.env.GEMINI_API_KEY,
        });

        this.vectorStore = null;
    }

    async loadAndProcessPDF(pdfPath) {
        try {
            if (!fs.existsSync(pdfPath)) {
                throw new Error(`PDF file not found at ${pdfPath}`);
            }

            const dataBuffer = fs.readFileSync(pdfPath);
            const pdfData = await pdf(dataBuffer);

            const processedPdfText = processContent(pdfData.text);

            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200,
            });

            const splitTexts = await textSplitter.createDocuments([processedPdfText]);

            this.vectorStore = await MemoryVectorStore.fromDocuments(
                splitTexts,
                this.embeddings
            );

            console.log('PDF processed and vectorized successfully');
        } catch (error) {
            console.error('Error processing PDF:', error);
            throw error;
        }
    }

    async queryPDF(query) {
        if (!this.vectorStore) {
            throw new Error('PDF not loaded. Call loadAndProcessPDF first.');
        }

        try {
            const processedQuery = processContent(query);

            const retriever = this.vectorStore.asRetriever(4);
            const relevantDocs = await retriever.getRelevantDocuments(processedQuery);

            const context = relevantDocs.map(doc => processContent(doc.pageContent)).join('\n\n');

            const prompt = `
            Context Information:
                ${context}
                
            Query: ${processedQuery}
            
            Instructions:
            1. Analyze the provided context carefully.
            2. Generate a concise, precise, and informative answer directly based on the context.
            3. Keep the response:
               - Strictly limited to information from the provided context.
               - Clear, easily understandable, and directly related to the query.
               
            Response Guidelines:
            - If the context does not contain sufficient information to answer the query:
               * Respond with: "I do not have enough information to answer this question."
               * Suggest: "Please ask a different question related to the available content."
               
            - If the query involves legal issues or deportation-related questions:
               * Acknowledge the issue and suggest seeking legal help if the context doesn't contain detailed information.
               * Use a response like: "It seems like you need legal assistance regarding deportation. Please consult a lawyer specializing in deportation matters."
               
            - If the query is asking about the chatbot itself, such as "What is this chatbot about?" or "What can this chatbot do?", and the context contains information about the chatbot’s purpose or functionality, provide a clear description of the chatbot's role or capabilities based on the context.
        
            Constraints:
            - Do not reference the source document.
            - Do not disclose any details about the document's origin.
            - Provide only the most relevant information.
              
            Response Format:
            [Concise Answer]
        `;
        
            
            const response = await this.llm.invoke(prompt);

            return {
                answer: processContent(response.content),
            };
        } catch (error) {
            console.error('Error querying PDF:', error);
            throw error;
        }
    }
}

const PDF_PATH = path.join(__dirname, 'faqs.pdf');
let pdfChatbot;

// Load PDF on Startup
(async () => {
    try {
        pdfChatbot = new PDFChatbot();
        await pdfChatbot.loadAndProcessPDF(PDF_PATH);
        console.log('PDF Processing Complete');
    } catch (error) {
        console.error('Critical Error during initial PDF processing:', error);
        process.exit(1);
    }
})();

// Dialogflow Integration
const sessionClient = new dialogflow.SessionsClient();
async function sendToDialogflow(message, sessionId) {
    try {
        if (!sessionId) throw new Error('Session ID is undefined.');
        if (!PROJECT_ID) throw new Error('Project ID is undefined.');

        const sessionPath = sessionClient.projectAgentSessionPath(
            PROJECT_ID,
            sessionId.toString()
        );

        const request = {
            session: sessionPath,
            queryInput: {
                text: {
                    text: message,
                    languageCode: 'en',
                },
            },
        };

        const [response] = await sessionClient.detectIntent(request);
        return response.queryResult;
    } catch (error) {
        console.error('Error in sendToDialogflow:', error.message);
        throw error;
    }
}

// Twilio Webhook
app.post('/twilio-webhook', async (req, res) => {
    console.log('Twilio Request:', req.body);

    try {
        const userMessage = req.body.Body;
        const fromNumber = req.body.From;

        const dialogflowResponse = await sendToDialogflow(userMessage, fromNumber);

        const dialogflowText = dialogflowResponse.fulfillmentText || 
                               'Sorry, I didn\'t understand that. Can you rephrase?';

        await twilioClient.messages.create({
            body: dialogflowText,
            from: process.env.TWILIO_WHATSAPP_NUMBER,
            to: fromNumber,
        });

        res.status(200).send('Message processed successfully');
    } catch (error) {
        console.error('Error processing Twilio request:', error);
        res.status(500).send('Error processing Twilio request');
    }
});

// Dialogflow Webhook
app.post('/dialogflow-webhook', async (req, res) => {
    console.log('Dialogflow Request:', JSON.stringify(req.body, null, 2));

    try {
        if (!pdfChatbot) {
            return res.status(503).send('Chatbot not initialized');
        }

        const agent = new WebhookClient({ request: req, response: res });
        const intentName = agent.intent || req.body.queryResult.intent.displayName;

        const handleWelcomeIntent = (agent) => {
            agent.add('Hi there! How can I assist you today?');
        };

        const handleFallbackIntent = async (agent) => {
            const userQuery = agent.query || req.body.queryResult.queryText || 'No query found';

            try {
                const result = await pdfChatbot.queryPDF(userQuery);
                agent.add(result.answer || 'Sorry, I couldn\'t find relevant information.');
            } catch (error) {
                console.error('Error processing PDF query:', error);
                agent.add('Sorry, I couldn\'t process your request right now.');
            }
        };

        const intentMap = new Map();
        intentMap.set('Default Welcome Intent', handleWelcomeIntent);
        intentMap.set('Default Fallback Intent', handleFallbackIntent);
        intentMap.set('PDF_Query_Intent', handleFallbackIntent);

        agent.handleRequest(intentMap);
    } catch (error) {
        console.error('Error processing Dialogflow request:', error);
        res.status(500).send('Internal Server Error');
    }
});

// Server Initialization
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

module.exports = app;

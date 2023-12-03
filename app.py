import os
import sys
from flask import Flask, render_template, request, jsonify, session
from openai import ChatCompletion
import secrets
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)   # Replace with your own secret key

# Initialize your Langchain and OpenAI code
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

@app.route("/", methods=["POST", "GET"])
def index():
    global chat_history

    if request.referrer and request.referrer.startswith(request.url_root):
        if request.method == "POST":
            query = request.form.get("query")
            selected_filter = request.form.get("filter")

            if query in ['quit', 'q', 'exit']:
                sys.exit()

            if selected_filter is None or selected_filter == "":
                return "Please select a filter.", 400

            # Store the selected filter in the session
            session['selected_filter'] = selected_filter  # Update selected_filter here

            filter_file_path = os.path.join("data", selected_filter)

            # if not os.path.exLogicoseists(filter_file_path):
            #     return "Selected filter does not exist.", 400

            with open(filter_file_path, "r") as data_file:
                data = data_file.read()

            prompt = f"I have the following data: {data}"
            response = ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a chatbot."},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": prompt}
                ]
            )

            datafiles = os.listdir("data/")    
            answer = response['choices'][0]['message']['content']
            pop = " What kind of information are you looking for?"

            chat_history.append((query, answer))
            return render_template("index.html", query=query, answer=answer, pop=pop, filters=[file for file in datafiles ])

    # Retrieve the selected filter from the session (if set)
    selected_filter = session.get('selected_filter', '')

    filters = os.listdir("data/")
    return render_template("index.html", filters=filters, selected_filter=selected_filter)



if __name__ == "__main__":
    app.run(debug=True)

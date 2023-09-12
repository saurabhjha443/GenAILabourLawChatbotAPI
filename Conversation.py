from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.chains import ConversationChain
# %%
from langchain.llms import GPT4All
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain


class LLMModelClass:

    def loadLLMmodel(self):
        loader = WebBaseLoader(["https://labour.gov.in/sites/default/files/code_on_wages_central_advisory_board_rules2021.pdf",
                                "https://labour.gov.in/sites/default/files/ir_gazette_of_india.pdf",
                                "https://labour.gov.in/sites/default/files/ss_code_gazette.pdf",
                                "https://labour.gov.in/sites/default/files/the_code_on_wages_2019_no._29_of_2019.pdf",
                                "https://blog.ipleaders.in/labour-laws/"])
        data = loader.load()

        # %%

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
        retriever = vectorstore.as_retriever()
        llm = GPT4All(

            model="C:/Users/saura/PycharmProjects/pythonProject/ggml-model-gpt4all-falcon-q4_0.bin",
            max_tokens=2048,

        )
        # %%
        # template = """You are a chatbot having a conversation with a human.
        # Use the following pieces of context to answer the question at the end.
        # If you don't know the answer, just say that you don't know, don't try to make up an answer.
        # Use three sentences maximum and keep the answer as concise as possible.
        # Always say 'thanks for asking!' at the end of the answer.
        # Given the following extracted parts of a long document and a question, create a final answer.
        #
        # {context}
        #
        # {chat_history}
        # Human: {human_input}
        # Chatbot:"""

        # prompt = PromptTemplate(
        #     input_variables=["chat_history", "human_input", "context"], template=template
        # )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # qa_chain = load_qa_chain(
        #     llm=llm, chain_type="stuff", memory=memory, prompt=prompt
        # )

        # memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

        # %%

        # QA_CHAIN_PROMPT = PromptTemplate(
        #   input_variables=["context", "question"],
        #   template=template,
        # )

        # Chain
        # chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

        # %%
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True, return_messages=True
        )

        # qa_chain = RetrievalQA.from_chain_type(
        #   llm,
        #   retriever=vectorstore.as_retriever(),
        #   chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        #  )
        return qa_chain

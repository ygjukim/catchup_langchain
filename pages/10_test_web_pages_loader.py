import bs4
from langchain_community.document_loaders import WebBaseLoader
import asyncio  # Import asyncio

async def load_and_process_document(page_url):  # Define an async function
    loader = WebBaseLoader(web_paths=[page_url])
    docs = []
    async for doc in loader.alazy_load(): # Now inside an async function
        docs.append(doc)

    assert len(docs) == 1
    doc = docs[0]

    print(f"{doc.metadata}\n")
    print(doc.page_content[:500].strip())
    return doc # Return the document if needed

async def main(): # Another async function to run the process
    page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
    doc = await load_and_process_document(page_url) # Await the function call
    # You can do something with the document here if needed.
    print(f"Document Title: {doc.metadata['title']}") # Example

if __name__ == "__main__":
    asyncio.run(main()) # Run the main async function
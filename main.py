import logging
from config import Config
from rag_system import RAGSystem
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        load_dotenv()
        config = Config()
        rag = RAGSystem(config)
        rag.setup()
        
        print("\nChat system ready! Type 'quit' to exit.")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in ['quit', 'exit', 'bye']: break
            if not query: continue
            print("\nAssistant: ", end="")
            print(rag.query(query))
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
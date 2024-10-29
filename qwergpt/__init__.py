from dotenv import load_dotenv
from pathlib import Path


project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'

load_dotenv(dotenv_path)

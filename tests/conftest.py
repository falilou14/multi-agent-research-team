"""
Chargé par pytest AVANT les modules testés.

Les modules du projet appellent load_dotenv() à l'import : si le .env local
active le tracing LangSmith, chaque test unitaire (faux modèle compris)
enverrait sa trace dans le projet LangSmith et le polluerait (vu au Ch.4 :
20 fausses traces en un lancement). load_dotenv() n'écrase pas une variable
déjà définie : la fixer ici suffit.
"""

import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

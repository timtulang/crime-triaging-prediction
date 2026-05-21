The actual code (Model Training) is located in the crimetriage.ipynb file. Designed to be ran on Google Colab for resource provisioning.

db.html contains a static dashboard for the results of the model and a tab for crime classification (VIOLENT or NON-VIOLENT).

TECHNICAL_TURNOVER.md contains technical information about the project. 


Important notes:
We pivoted from a model that predicts where a crime might appear to a triaging approach. This is done for better resource allocation.

Say a report is predicted to be NON-VIOLENT; authorities can allocate less manpower in that area. The opposite is true for VIOLENT crimes.


HOW TO RUN:
- CD to this project folder.
- Run "pip install -r requirements.txt" to install dependencies
- Run "uvicorn server:app --reload"
- Run html file using "live server" extension in VS Code


FOR CHRIS:
nasa db yung results if u need images, pero yung pso_results.json naman if u need actual text file
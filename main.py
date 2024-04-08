from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import joblib

class PredictStudentResult(Action):
    def name(self) -> Text:
        return "action_predict_student_result"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Load the trained model
        model = joblib.load('student_result_predictor.joblib')

        # Get the required inputs from the user's message
        writing_score = tracker.latest_message['entities'][0]['writing_score']
        reading_score = tracker.latest_message['entities'][0]['reading_score']
        parental_education = tracker.latest_message['entities'][0]['parental_education']

        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'writing score': [writing_score],
            'reading score': [reading_score],
            'parental level of education_' + parental_education: [1]
        })

        # Make the prediction
        predicted_score = model.predict(input_data)[0]

        # Send the predicted score as the bot's response
        dispatcher.utter_message(text=f"Based on the given scores, the predicted math score is {predicted_score}.")

        return []

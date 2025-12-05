import json
import os

class MentalHealthRecommender:
    def __init__(self, resource_file='data/resources.json'):
        # Load the resources database
        base_path = os.path.dirname(os.path.dirname(__file__)) # Go up one level
        full_path = os.path.join(base_path, resource_file)
        
        with open(full_path, 'r') as f:
            self.resources = json.load(f)

    def get_recommendations(self, risk_label, user_features=None):
        """
        risk_label: 'Low', 'Moderate', or 'High'
        user_features: Dict of specific metrics (e.g. {'sleep': 4})
        """
        # 1. Get General Advice based on Model Prediction
        base_advice = self.resources['risk_levels'].get(risk_label, {})
        
        output = {
            "alert": base_advice.get("message"),
            "type": base_advice.get("alert_type"),
            "resources": base_advice.get("resources",),
            "tips":
        }

        # 2. Check Specific Triggers (Rule-Based Layer)
        if user_features:
            triggers = self.resources['specific_triggers']
            
            # Example Logic: Check Sleep
            if user_features.get('Sleep Duration') is not None:
                # Simple parsing logic (assuming input might be "5 hours")
                try:
                    sleep_val = float(str(user_features).split())
                    if sleep_val < 5:
                        output['tips'].append(triggers['sleep_deprived']['advice'])
                except:
                    pass # Handle parsing errors gracefully

            # Example Logic: Check Pressure
            if user_features.get('Academic Pressure', 0) >= 4:
                output['tips'].append(triggers['academic_stress']['advice'])

        return output
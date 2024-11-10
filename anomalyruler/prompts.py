def derive_normal_activity(frame_descriptions, activity_amount):
        prompt = f"""Given the frame descriptions {frame_descriptions}, please list at most {activity_amount} unique human activities from these frame descriptions.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        Example:
        1. Walking
        2. Standing
        
        Answer:
        """
        return prompt

def derive_normal_objects(frame_descriptions, object_amount):
        prompt = f"""Given the frame descriptions {frame_descriptions}, please list at most {object_amount} unique environmental objects from these frame descriptions.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        Example: 
        1. Pathway
        2. Building
        
        Answer:
        """
        return prompt

def derive_abnormal_activities(normal_activities, activity_amount):
        prompt = f"""Given these normal activities {normal_activities}, please list at most {activity_amount} potential abnormal human activities from the context of these normal activites.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        Example:
        1. Fighting
        2. Running

        Answer:
        """
        return prompt

def derive_abnormal_objects(normal_objects, object_amount):
        prompt = f"""Given these normal objects {normal_objects}, please list at most {object_amount} potential abnormal environmental objects from the context of these normal objects.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        Example:
        1. Car
        2. Weapon
        
        Answer:
        """
        return prompt
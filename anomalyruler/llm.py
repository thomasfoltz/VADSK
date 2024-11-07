from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    return tokenizer, model

def parse_output(output, limit):
        parsed_output, inside_block = [], False
        for line in output.split('\n'):
            if 'START' in line:
                inside_block = True
                continue
            if 'END' in line:
                inside_block = False
                continue
            line = line.strip()[3:]
            if inside_block and line!='':
                parsed_output.append(line)
        return parsed_output[4:4+limit]

def derive_normal_activity(frame_descriptions, activity_amount):
        prompt = f"""Given the frame descriptions {frame_descriptions}, please list {activity_amount} unique human activities from these frame descriptions.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        START
        Human Activities:
        1. 
        2. 
        END
        """
        return prompt

def derive_normal_objects(frame_descriptions, object_amount):
        prompt = f"""Given the frame descriptions {frame_descriptions}, please list {object_amount} unique environmental objects from these frame descriptions.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        START
        Environmental Objects:
        1.
        2.
        END
        """
        return prompt

def derive_abnormal_activities(normal_activities, activity_amount):
        prompt = f"""Given these normal activities {normal_activities}, please list {activity_amount} potential abnormal human activities.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        START
        Abnormal Human Activities:
        1. 
        2. 
        END
        """
        return prompt

def derive_abnormal_objects(normal_objects, object_amount):
        prompt = f"""Given these normal objects {normal_objects}, please list {object_amount} potential abnormal environmental objects.
        List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
        START
        Abnormal Environmental Objects:
        1. 
        2. 
        END
        """
        return prompt

def generate_output(input, tokenizer, model, output_limit=256):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True).to("cuda")
    attention_mask = inputs['attention_mask']
    input_ids_length = len(inputs['input_ids'][0])
    output = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=input_ids_length+output_limit
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output
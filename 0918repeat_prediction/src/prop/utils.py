import re


def gen_problem(record, tokenizer):
    comp_name = record["CompName"]
    smiles = record["SMILES"]
    unit = record["unit"]
    property_name = record["Property"]
    q = f"""Predict the {property_name} {unit} of the following compound. 
    #Restriction: Output must contain "#Reason" section which explains the step-by-step quantitative reasoning behind the prediction. Output must also contain "#Predicted" section which contains only the predicted value (number only).
    There must not be inconsistency between the predicted value and the reasoning.
    #Name: {comp_name}
    #SMILES: {smiles}"""
    actual_value = record["Value"]

    chat = [
        {"role": "user", "content": q},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False,)
    assist_prefix = "assistant\n\n#Reason\n"
    prompt += assist_prefix

    return prompt, actual_value


def gen_reason(record, tokenizer):
    comp_name = record["CompName"]
    smiles = record["SMILES"]
    unit = record["unit"]
    property_name = record["Property"]
    actual_value = record["Value"]
    q = f"""Provide the quantitative Reason and Prediction so that a scientist,
who does not know the {property_name} {unit} of {comp_name} ({smiles}), can predict the value.

#Commands
focusing on each functional groups.
- Actual value and Prediction must match each other.
- Your answer must contain #Reason and #Prediction section.
- Never include the actual value ({actual_value} {unit}) in #Reason, but include it in #Prediction.
- #Prediction must contain only the predicted value (number only).
"""

    chat = [
        {"role": "user", "content": q},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False,)
    assist_prefix = "assistant\n\n#Reason\n"
    prompt += assist_prefix

    return prompt, actual_value


def parse_prediction(problems, predicted_text_list):
    n_records = len(problems)
    # parse
    good_records = []
    for i in range(n_records):
        problems[i]["CompoundName"] = problems[i]["record"]["CompName"]
        problems[i]["SMILES"] = problems[i]["record"]["SMILES"]
        problems[i]["Property"] = problems[i]["record"]["Property"]
        problems[i]["Unit"] = problems[i]["record"]["unit"]
        problems[i]["predicted"] = predicted_text_list[i].split(
            "#Prediction\n")[-1]
        problems[i]["predicted_text"] = predicted_text_list[i]
        # 誤差率
        problems[i]["actual_value"] = float(problems[i]["actual_value"])
        try:
            error = abs((problems[i]["actual_value"]-float(problems[i]
                        ["predicted"]))/problems[i]["actual_value"])
            problems[i]["error_rate"] = error
            good_records.append(problems[i])
        except:
            continue
    return good_records


def parse_prediction_with_check(problems, masked_reason_list, predicted_value_list):
    n_records = len(problems)
    # parse
    good_records = []
    for i in range(n_records):
        problems[i]["CompoundName"] = problems[i]["record"]["CompName"]
        problems[i]["SMILES"] = problems[i]["record"]["SMILES"]
        problems[i]["Property"] = problems[i]["record"]["Property"]
        problems[i]["Unit"] = problems[i]["record"]["unit"]
        problems[i]["predicted"] = predicted_value_list[i]
        problems[i]["reason"] = masked_reason_list[i]
        # 誤差率
        problems[i]["actual_value"] = float(problems[i]["actual_value"])
        try:
            error = abs((problems[i]["actual_value"]-float(problems[i]
                        ["predicted"]))/problems[i]["actual_value"])
            problems[i]["error_rate"] = error
            good_records.append(problems[i])
        except:
            continue
    return good_records


def gen_masked_reason(predicted_text_list, problems, i):
    predicted_reason = predicted_text_list[i]
    smiles = problems[i]["record"]["SMILES"]
    comp_name = problems[i]["record"]["CompName"]
    actual_value = problems[i]["actual_value"]

    masked_reason = predicted_reason.replace(comp_name, "Compound X")
    masked_reason = masked_reason.replace(comp_name.lower(), "Compound X")
    masked_reason = masked_reason.replace(str(smiles), "*")
    masked_reason = masked_reason.replace(str(actual_value), "*")
    masked_reason = masked_reason.split("#Prediction")[0].strip()
    masked_reason = re.sub(
        r'Compound X \([^)]*\)', 'Compound X', masked_reason)

    return masked_reason


def gen_masked_prediction_problem_prompts(predicted_text_list, problems, tokenizer):
    n_records = len(problems)
    masked_reasons = [gen_masked_reason(
        predicted_text_list, problems, i) for i in range(n_records)]

    prompt_template = """Predict the material property of Compound X. Your output consists of only #Prediction section. Only number is allowed in the #Prediction section. Do not include any other information in the #Prediction section."""
    prompt_list = []
    for i in range(n_records):
        chat = [
            {"role": "user", "content": prompt_template +
                "\n #Reason\n"+masked_reasons[i]},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False,)
        prompt = prompt + "assistant\n\n#Prediction\n"
        prompt_list.append(prompt)
    return prompt_list, masked_reasons
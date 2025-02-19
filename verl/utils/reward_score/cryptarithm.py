import re
import random
import torch

def extract_solution(solution_str):
    # Countdown.py removes string before "Assistant: "

    answer_pattern = r'<answers>\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\s*</answers>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        matches = matches[-1]
        final_answer = (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
    else:
        final_answer = None

    return final_answer

def validate_equation(equation, ground_encryption):
    if (equation[0] + equation[1] + equation[2] != equation[3]):
        return False

    sol_map = torch.zeros(26)

    for i, num in enumerate(equation):
        encrypt_num = ground_encryption[i]

        if (len(num) != len(encrypt_num[i])):
            return False
        
        for j, digit in enumerate(num):
            encrypt_digit = encrypt_num[j]

            if (sol_map[encrypt_digit] == 0):
                sol_map[encrypt_digit] = digit
            elif (sol_map[encrypt_digit] != digit):
                return False
    return True

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    do_print = random.randint(1, 100) == 1
    
    ground_encryption = ground_truth['encryption']
    if do_print:
        print(f"--------------------------------")
        print(f"Ground Truth: {ground_truth[0]}+{ground_truth[1]}+{ground_truth[2]}={ground_truth[3]}")

    answer = extract_solution(solution_str)

    if answer is None:
        if do_print:
            print(f"Wrong Format")
        return 0
    
    if validate_equation(answer, ground_encryption):
        print(f"Correct Answer: {answer[0]}+{answer[1]}+{answer[2]}={answer[3]}")
        return score
    else:
        if do_print:
            print(f"Wrong Answer, Correct Format")
        return format_score
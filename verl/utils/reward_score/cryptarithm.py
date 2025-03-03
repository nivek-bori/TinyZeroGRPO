import re
import random
import torch

def extract_solution(solution_str):
    # Countdown.py removes string before "Assistant: "
    solution_str = solution_str.split("Assistant:", 1)[1]

    answer_pattern = r'<answers>\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\s*</answers>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        matches = matches[-1]
        return (match.group(1), match.group(2), match.group(3), match.group(4))
    else:
        return None

def validate_equation(equation, ground_encryption):
    if sum(map(int, equation[:3])) != int(equation[3]):
        return False

    sol_map = torch.zeros(26, dtype=torch.int)

    for num, enc_num in zip(equation, ground_encryption):
        if len(num) != len(enc_num):
            return False

        for digit_char, enc_digit in zip(num, enc_num):
            digit_val = int(digit_char)
            if sol_map[enc_digit] == 0:
                sol_map[enc_digit] = digit_val
            elif sol_map[enc_digit] != digit_val:
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

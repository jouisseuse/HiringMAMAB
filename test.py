# 定义公司和候选人数量
n_firms = 4
n_candidates = 4

# 初始化每个 candidate 的雇佣统计结构
hire_summary_candidate = {
    candidate_id: {
        'id': candidate_id,
        'firms': {firm_id: {'hire_count': 0, 'not_hire_count': 0} for firm_id in range(1, n_firms + 1)}
    }
    for candidate_id in range(1, n_candidates + 1)
}

# 初始化每个 firm 的收益统计结构
hire_summary_firm = {
    firm_id: {
        'id': firm_id,
        'candidates': {candidate_id: {'rewards': []} for candidate_id in range(1, n_candidates + 1)}
    }
    for firm_id in range(1, n_firms + 1)
}

# 示例数据
candidate_results = {
    1: [1, 1],
    2: [2, 0],
    3: [3, 1],
    4: [4, 0]
}

firm_results = {
    1: [1, 100],
    2: [2, 0],
    3: [3, 50],
    4: [4, 0]
}

# 遍历 candidate_results 并更新 hire_summary_candidate 和 hire_summary_firm
for candidate_id, (firm_id, hire_status) in candidate_results.items():
    # 更新 candidate 的雇佣统计
    if hire_status == 1:
        hire_summary_candidate[candidate_id]['firms'][firm_id]['hire_count'] += 1
    else:
        hire_summary_candidate[candidate_id]['firms'][firm_id]['not_hire_count'] += 1
    
    # 更新 firm 的收益统计
    reward = firm_results[firm_id][1]
    hire_summary_firm[firm_id]['candidates'][candidate_id]['rewards'].append(reward)

# 遍历 hire_summary_candidate
print("Candidate Hire Summary:")
for candidate_id, summary in hire_summary_candidate.items():
    print(f"Candidate {summary['id']}:")
    for firm_id, stats in summary['firms'].items():
        print(f"  Firm {firm_id} - Hired: {stats['hire_count']} times, Not Hired: {stats['not_hire_count']} times")

# 遍历 hire_summary_firm
print("\nFirm Rewards Summary:")
for firm_id, summary in hire_summary_firm.items():
    print(f"Firm {summary['id']}:")
    for candidate_id, stats in summary['candidates'].items():
        print(f"  Candidate {candidate_id} - Rewards: {stats['rewards']}")
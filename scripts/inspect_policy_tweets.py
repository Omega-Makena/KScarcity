import pandas as pd

df = pd.read_csv('data/synthetic_kenya_policy/tweets.csv')
print('Shape:', df.shape)
print('\nColumns:', list(df.columns))

print('\nNew fields sample:')
policy = df[df['policy_event_id'].notna()]
print(f'  Policy tweets: {len(policy)} / {len(df)} ({len(policy)*100//len(df)}%)')
print(f'  Unique events: {policy.policy_event_id.nunique()}')
print(f'  Phases: {policy.policy_phase.value_counts().to_dict()}')
print(f'  Sentiment range: [{policy.sentiment_score.min():.3f}, {policy.sentiment_score.max():.3f}]')
print(f'  Stance distribution:')
for s, label in [(-1.0, 'anti'), (0.0, 'neutral'), (1.0, 'pro')]:
    n = (policy.stance_score == s).sum()
    print(f'    {label}: {n} ({n*100//len(policy)}%)')
print(f'  Topic clusters: {policy.topic_cluster.value_counts().to_dict()}')

print('\nSample policy tweets:')
for _, row in policy.sample(8, random_state=42).iterrows():
    label = 'anti' if row.stance_score < 0 else ('pro' if row.stance_score > 0 else 'neutral')
    print(f'  [{row.policy_event_id}|{row.policy_phase}|{label}] {row.text[:140]}')

print('\nSample organic (non-policy) tweets:')
organic = df[df['policy_event_id'].isna()].sample(5, random_state=42)
for _, row in organic.iterrows():
    print(f'  [{row.intent}] {row.text[:140]}')

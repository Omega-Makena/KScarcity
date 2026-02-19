import pandas as pd

df = pd.read_csv('data/synthetic_kenya_policy/tweets.csv')

# Sheng markers (urban Kenyan slang)
sheng = ['bana','maze','wasee','msee','noma','safi','poa','manze','walai',
         'fiti','sawa','mbaya','legit','manze','buda','mathe','kali',
         'mazematics','iko aje','si poa','mambo','rada','shwari','anguka nayo']

# Swahili markers
swahili = ['lazima','sisi','wakenya','serikali','ushuru','kodi','bei',
           'mshahara','stima','sukari','matibabu','hospitali','katiba',
           'usalama','wananchi','tunakufa','hatuwezi','tumechoka','pamoja',
           'maandamano','hakuna','sasa','mwisho','kila','bunge','nyumba',
           'mfuko','wafanyikazi','elimu','mkulima','jeshi','kaunti',
           'bajeti','umeme','treni','nauli','bima','afya']

# Mixed code-switching patterns
codeswitching = ['hii ni','watu wame','mambo ni','vitu kwa ground',
                 'tunasema','haiko sawa','inaendelea','wanatucheza',
                 'wamekula','hatutaki','tunaunga','tunaangalia',
                 'imetosha','amka','toka','jitokeze','sambaza']

text_lower = df['text'].str.lower()

sheng_mask = text_lower.str.contains('|'.join(sheng), na=False, regex=True)
swahili_mask = text_lower.str.contains('|'.join(swahili), na=False, regex=True)
cs_mask = text_lower.str.contains('|'.join(codeswitching), na=False, regex=True)
any_mask = sheng_mask | swahili_mask | cs_mask

total = len(df)
print(f'Total tweets: {total:,}')
print(f'Sheng:          {sheng_mask.sum():>6,} ({sheng_mask.sum()*100//total}%)')
print(f'Swahili:        {swahili_mask.sum():>6,} ({swahili_mask.sum()*100//total}%)')
print(f'Code-switching: {cs_mask.sum():>6,} ({cs_mask.sum()*100//total}%)')
print(f'Any non-English:{any_mask.sum():>6,} ({any_mask.sum()*100//total}%)')
print(f'English-only:   {(~any_mask).sum():>6,} ({(~any_mask).sum()*100//total}%)')

print('\n=== Sample SHENG tweets ===')
for _, row in df[sheng_mask].sample(6, random_state=7).iterrows():
    print(f'  {row.text[:160]}')

print('\n=== Sample SWAHILI tweets ===')
for _, row in df[swahili_mask].sample(6, random_state=7).iterrows():
    print(f'  {row.text[:160]}')

print('\n=== Sample CODE-SWITCHING (Sheng+English mix) ===')
both = sheng_mask & swahili_mask
if both.sum() > 0:
    for _, row in df[both].sample(min(6, both.sum()), random_state=7).iterrows():
        print(f'  {row.text[:160]}')
else:
    mixed = cs_mask & (sheng_mask | swahili_mask)
    for _, row in df[mixed].sample(min(6, mixed.sum()), random_state=7).iterrows():
        print(f'  {row.text[:160]}')

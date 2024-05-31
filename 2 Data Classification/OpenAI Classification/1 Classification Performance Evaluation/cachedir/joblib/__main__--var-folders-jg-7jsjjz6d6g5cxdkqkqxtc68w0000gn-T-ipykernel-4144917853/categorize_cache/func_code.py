# first line: 1
@memory.cache
def categorize_cache(row):
    prompt = """
    Below is the text to a review. Classify it as one or more of the following categories:

    - l1_inaccurate_cycle_prediction: This category suggests that the app's cycle prediction algorithm is inaccurate, sometimes leading to unplanned pregnancies.
    - l2_delayed_customer_service: This category suggests that difficulty in contacting customer service and long wait times, which oftentimes result in late or inaccurate deliveries of prescriptions and medications.
    - l3_poor_prescription_management: This category suggests users experience issues such as missing or incorrect prescriptions, incorrect birth control medications, inaccurate refill frequencies, late deliveries, and canceled medications.
    - l4_problematic_billing_practices: This category suggests that users encounter unexpected charges including but not limited to auto-renewals without notification, and charges on old credit cards without refunds, or they fail to use the current insurance plan for insurance billing.

    The review may be assigned to multiple categories. Please list all applicable categories based on the review content.

    Review text:

    {text}
    """
    
    results = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a review assistant. Be brief in your responses."},
            {"role": "user", "content": prompt.format(text=row['review'])}
        ],
        temperature=0,
        logprobs=True
    )

    return pd.Series({
        'content': results.choices[0].message.content,
        'probability': math.exp(results.choices[0].logprobs.content[0].logprob)
    })

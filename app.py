import streamlit as st
import pandas as pd
import re
from openai import OpenAI

# --- Helper Functions ---

def clean_text(text):
    """Removes quotes, citations, and extra whitespace."""
    text = re.sub(r'["“””-]', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text.strip()

def openai_gpt_request(api_key, prompt, model="gpt-4o"): # or "gpt-4"
    """Makes a request to the OpenAI GPT-4 API."""
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be precise and follow instructions EXACTLY."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Adjust for creativity (0.0 is deterministic, 1.0 is most creative)
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error from OpenAI API: {e}")
        return None

def generate_content(api_key, company_name, keywords, debug_mode=False):
    """Generates content for each keyword, using OpenAI GPT-4."""
    all_posts = []
    keyword_batches = [keywords[i:i + 1] for i in range(0, len(keywords), 1)]  # Batch size of 1 (or adjust)

    for batch in keyword_batches:
        for keyword in batch: #now we can simply iterate
            prompt = f"""
Context: You are generating content for a new internal Intrafeed at {company_name}. Employees won't contribute if they don't see content, so you must 'seed' the platform.

ABSOLUTE OUTPUT FORMAT REQUIREMENTS:
- Output MUST be plain text.
- Output MUST consist of EXACTLY ONE line.
- The line MUST contain the following fields, separated by '|||': Title, Post Body, Comment 1, Comment 2, Comment 3, Comment 4, Comment 5
- There MUST be NO other text before, after, or between the data fields. NO headers. NO explanations. NO newlines within a field.
- Example: Title|||Post Body|||Comment 1|||Comment 2|||Comment 3|||Comment 4|||Comment 5

Directives:
- Create one post based on the keyword: **{keyword}**
- Post should not look AI generated and should mimic a reddit style post
- Include the keyword in the Title, Post Body, and all Comments.
- Comments should be atleast 150 words aim for that 
- The post should be in a Reddit style.
- Some posts should have typos or informal language (like Hinglish if {company_name} is Indian).
- The post should NOT look AI-generated.
- Add specific details like location (cities where {company_name} operates), {company_name}'s tools, branch names, etc.
- Vary the character length of the Title, Post Body, and each Comment.
- For comments, aim for a mix of lengths: some short, some medium, some longer (around 20-250 characters).
- Don't overuse emojis.
- Base content on reviews from AmbitionBox, Glassdoor, Fishbowl, but rephrase to avoid direct copying.
- Do not hallucinate. Provide only factual information, or phrase as a question.
- Be specific enough for employees to recognize it as internal, but ensure no false information.
- Posts should be anonymous. No usernames. No hashtags.
- Comments must vary in tone, style, and content.
"""
            response_text = openai_gpt_request(api_key, prompt)
            if response_text:
                if debug_mode:
                    print(f"----- Raw Response for keyword '{keyword}': -----\n{response_text}")
                all_posts.append(response_text)

    return all_posts



def parse_responses(responses, debug_mode=False):
    """Parses responses, handling variable numbers of comments."""
    all_dfs = []
    for line in responses:
        if debug_mode:
            print(f"----- Processing Line: -----\n{line}")

        parts = re.split(r'\s*\|\|\|\s*', line)
        parts = [p for p in parts if p]  # Remove empty strings

        if debug_mode:
            print(f"----- Split Parts: {parts}")

        if len(parts) >= 2:
            data = {
                'Title': [clean_text(parts[0])],
                'Post Body': [clean_text(parts[1])],
                'Comment 1': [clean_text(parts[2]) if len(parts) > 2 else ""],
                'Comment 2': [clean_text(parts[3]) if len(parts) > 3 else ""],
                'Comment 3': [clean_text(parts[4]) if len(parts) > 4 else ""],
                'Comment 4': [clean_text(parts[5]) if len(parts) > 5 else ""],
                'Comment 5': [clean_text(parts[6]) if len(parts) > 6 else ""],
            }
            df = pd.DataFrame(data)
            all_dfs.append(df)
        else:
            st.warning(f"Too few fields ({len(parts)}) in line. Skipping.\nLine Snippet: {line[:100]}...")

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# --- Streamlit App ---

st.title("SEO Content Generator | Built at Grapevine")

# Use session state to store the API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# Input for API key (only if not already set)
if not st.session_state.api_key:
    st.write("Please enter your OpenAI API key to use this app.")
    st.session_state.api_key = st.text_input("OpenAI API Key:", type="password")

# Only show the main app if the API key is set
if st.session_state.api_key:
    debug_mode = st.checkbox("Enable Debug Mode")
    company_name = st.text_input("Enter the company name:", value="IBM India")
    keyword_input = st.text_area("Enter keywords (comma-separated, up to 30):")
    keywords = [k.strip() for k in re.split(r'[,\n]', keyword_input) if k.strip()]

    if len(keywords) > 30:
        st.warning("Max 30 keywords.")
        keywords = keywords[:30]

    if st.button("Generate Content"):
        if not company_name or not keywords:
            st.error("Please enter company name and keywords.")
        else:
            with st.spinner("Generating..."):
                progress_bar = st.progress(0)
                raw_responses = generate_content(st.session_state.api_key, company_name, keywords, debug_mode)
                progress_bar.progress(50)
                if raw_responses:
                    df = parse_responses(raw_responses, debug_mode)
                    progress_bar.progress(100)
                    if not df.empty:
                        st.dataframe(df)
                        csv = df.to_csv(index=False)
                        st.download_button("Download CSV", csv, 'intrafeed_content.csv', 'text/csv')
                    else:
                        st.error("No content generated successfully.")
                else:
                    st.error("No content generated. Check API Key/prompt.")

else:
    st.write("Enter your API key to begin.")

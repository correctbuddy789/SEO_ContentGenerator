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


def perplexity_sonar_request(api_key, prompt, model="sonar-pro"):
    """Makes a request to the Perplexity Sonar API."""
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be precise and follow instructions EXACTLY."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error from Perplexity API: {e}")
        return None


def generate_content(api_key, company_name, keywords, debug_mode=False):
    """Generates content with a simplified loop-based prompt."""
    all_posts = []
    keyword_batches = [keywords[i:i + 3] for i in range(0, len(keywords), 3)]

    for batch in keyword_batches:
        prompt = f"""
Context: You are generating content for a new internal Intrafeed at {company_name}.

ABSOLUTE OUTPUT FORMAT REQUIREMENTS:
- Output MUST be plain text.
- For EACH keyword, create ONE SEPARATE line of output.
- EACH line MUST contain the following fields, separated by '|||': Title, Post Body, Comment 1, Comment 2, Comment 3, Comment 4, Comment 5
- NO headers, explanations, or newlines WITHIN a field.

Directives:
- Reddit style posts.
- Some typos/informal language.
- Add {company_name} details.
- Vary comment lengths. Aim for 500-700 characters for comments.
- Few emojis.
- Based on reviews, but rephrase.
- Factual or questions.
- Specific but no false info.
- Anonymous posts.
- DETAILED and ELABORATE comments.

INSTRUCTION LOOP:
"""
        for i, keyword in enumerate(batch):
            prompt += f"""
--- Keyword {i+1}: {keyword} ---
Generate ONE line for '{keyword}': Title|||Post Body|||Comment 1|||Comment 2|||Comment 3|||Comment 4|||Comment 5
"""

        response_text = perplexity_sonar_request(api_key, prompt)
        if response_text:
            if debug_mode:
                print(f"----- Raw Response for batch {batch}: -----\n{response_text}")

            for line in response_text.splitlines():
                line = line.strip()
                if line:
                    all_posts.append(line)

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

        # Handle cases with fewer than 7 parts (fewer than 5 comments)
        if len(parts) >= 2:  # Need at least Title and Post Body
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

st.title("SEO Content Generator | Built at Grapevine ")

# Use session state to store the API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# Input for API key (only if not already set)
if not st.session_state.api_key:
    st.write("Please enter your Perplexity API key to use this app.")
    st.session_state.api_key = st.text_input("Perplexity API Key:", type="password")

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
    st.write("Enter your API key to begin.") #Initial message

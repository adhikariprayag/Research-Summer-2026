# GPT Resume Entity Reconstruction Prompt

You are analyzing plain text extracted from a resume. The text may be messy because the original resume may have used multiple columns, sidebars, images, OCR, unusual formatting, or broken reading order.

Your task is to extract and reconstruct resume entities using only the information present in the provided resume text.

Extract the following entity types:

- PERSON: names of people, including applicant names and named references
- ORG: companies, universities, schools, institutions, organizations, or formal groups
- DATE: dates, years, date ranges, graduation dates, job dates, and project dates
- GPE: cities, states, countries, and geographic locations
- SKILL: technical skills, software tools, programming languages, frameworks, platforms, design tools, IT systems, and professional skills

Important rules:

1. Use only information supported by the resume text.
2. Do not invent entities that are not present.
3. If OCR creates an obvious spelling mistake, reconstruct the likely intended entity only when the evidence is clear.
4. Do not include section headings such as Summary, Education, Experience, Skills, Projects, or Profile unless they are part of a real entity.
5. Do not include page markers such as "--- Page 1 ---".
6. Do not include duplicate entities. If the same entity appears multiple times in the resume, include it only once.
7. Do not repeat the same year, date, skill, organization, or name multiple times.
8. Limit the final output to the most important resume entities. Prefer quality over quantity.
9. Keep entity text concise.
10. Return only valid JSON.
11. Do not include explanations, markdown, or comments in the final answer.

Return the answer in exactly this JSON structure:

{
"resume_id": "RESUME_ID_HERE",
"entities": [
{
"entity_text": "Example Entity",
"entity_type": "ORG"
}
]
}

Allowed entity_type values are only:

PERSON
ORG
DATE
GPE
SKILL

Resume ID:
RESUME_ID_HERE

Resume text:
RESUME_TEXT_HERE

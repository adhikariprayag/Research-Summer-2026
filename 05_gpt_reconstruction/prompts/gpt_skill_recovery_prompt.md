# GPT Resume Skill Recovery Prompt

You are analyzing plain text extracted from a Computer Science or Information Technology resume.

The text may be messy because of PDF extraction, OCR, columns, sidebars, tables, or broken reading order.

Your task is to extract only SKILL entities that are explicitly supported by the resume text.

Include:

- programming languages
- frameworks and libraries
- databases
- cloud platforms and services
- operating systems
- developer tools
- networking technologies
- cybersecurity tools and concepts
- data engineering and machine learning technologies
- enterprise software
- technical methodologies
- multi-word technical skills and technical phrases

Important rules:

1. Use only information present in the resume text.
2. Do not invent or infer unsupported skills.
3. Do not extract PERSON, ORG, DATE, or GPE entities.
4. Do not include job titles, degrees, schools, employers, locations, or section headings.
5. Do not include vague soft skills.
6. Preserve meaningful multi-word skills when they appear in the text.
7. Include both the full technical phrase and its abbreviation only when both explicitly appear.
8. Remove duplicate skills.
9. Keep the original wording when possible.
10. Return only valid JSON with no markdown or explanation.

Return exactly this structure:

{
  "resume_id": "RESUME_ID_HERE",
  "skills": [
    {
      "entity_text": "Example Technical Skill",
      "entity_type": "SKILL"
    }
  ]
}

Resume ID:
RESUME_ID_HERE

Resume text:
RESUME_TEXT_HERE

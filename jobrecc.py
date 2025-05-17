from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from collections import defaultdict

model = SentenceTransformer('paraphrase-albert-small-v2')

with open('job_roles_batch.json', 'r') as f:
    jobs_data = json.load(f)

job_embeddings = {}
skill_to_jobs = defaultdict(list)

for job in jobs_data:
    job_text = f"{job['role']} requires skills like {' '.join(job['essential_skills'])}"
    job_embedding = model.encode(job_text)
    job_embeddings[job['role']] = {
        'embedding': job_embedding,
        'skills': job['essential_skills'],
        'resources': job['learning_resources']
    }
    for skill in job['essential_skills']:
        skill_to_jobs[skill].append(job['role'])

# Normalization mapping same as your code
def normalize_skills(skills):
    """Normalize skill names for better matching"""
    skill_mapping = {
        # Programming Languages
        'python': 'Python',
        'java': 'Java',
        'javascript': 'JavaScript',
        'js': 'JavaScript',
        'c++': 'C++',
        'c#': 'C#',
        'dart': 'Dart',
        'kotlin': 'Kotlin',
        'solidity': 'Solidity',
        'HTML':'HTML',
        'HTML5':'HTML',
        'CSS':'CSS',
        'CSS3':'CSS',

        # Libraries/Frameworks
        'tensorflow': 'TensorFlow',
        'pytorch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'scikit': 'scikit-learn',
        'scikit learn': 'scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'react': 'React',
        'redux': 'Redux',
        'node': 'Node.js',
        'nodejs': 'Node.js',
        'express': 'Express',
        'django': 'Django',
        'flutter': 'Flutter',
        'web3': 'Web3.js',
        'web3js': 'Web3.js',
        'web3.js': 'Web3.js',
        'opencv': 'OpenCV',
        'spacy': 'SpaCy',

        # Concepts
        'dl': 'Deep Learning',
        'deep learning': 'Deep Learning',
        'ml': 'Machine Learning',
        'machine learning': 'Machine Learning',
        'ai': 'Artificial Intelligence',
        'nlp': 'NLP',
        'dsa':'Data Structures',
        'dsa':'Algorithms',
        'DSA':'Data Structures',
        'data structures': 'Data Structures',
        'algorithms': 'Algorithms',
        'data science': 'Data Science',
        'natural language processing': 'NLP',
        'cnn': 'CNN',
        'convolutional neural network': 'CNN',
        'rest': 'REST APIs',
        'rest api': 'REST APIs',
        'restful': 'REST APIs',
        'api': 'REST APIs',
        'cicd': 'CI/CD',
        'ci/cd': 'CI/CD',
        'continuous integration': 'CI/CD',
        'continuous deployment': 'CI/CD',
        'etl': 'ETL Pipelines',
        'extract transform load': 'ETL Pipelines',
        'big data': 'BigQuery',
        'bigquery': 'BigQuery',
        'gcp': 'BigQuery',

        # Tools/Platforms
        'aws': 'AWS',
        'amazon web services': 'AWS',
        'azure': 'Azure',
        'microsoft azure': 'Azure',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'k8s': 'Kubernetes',
        'jenkins': 'Jenkins',
        'terraform': 'Terraform',
        'iac': 'Infrastructure as Code',
        'infrastructure as code': 'Infrastructure as Code',
        'jira': 'JIRA',
        'atlassian': 'JIRA',
        'figma': 'Figma',
        'adobe xd': 'Adobe XD',
        'xd': 'Adobe XD',
        'unity': 'Unity',
        'android': 'Android Studio',
        'android studio': 'Android Studio',
        'firebase': 'Firebase',
        'mongodb': 'MongoDB',
        'mysql': 'MySQL',
        'postgres': 'PostgreSQL',
        'postgresql': 'PostgreSQL',
        'huggingface': 'HuggingFace',

        # Methodologies
        'agile': 'Agile Methodologies',
        'scrum': 'Scrum',
        'kanban': 'Kanban',

        # Other Skills
        'git': 'Git',
        'github': 'Git',
        'gitlab': 'Git',
        'sql': 'SQL',
        'nosql': 'SQL',
        'html': 'HTML',
        'html5': 'HTML',
        'css': 'CSS',
        'css3': 'CSS',
        'linux': 'Linux',
        'unix': 'Linux',
        'bash': 'Shell Scripting',
        'shell': 'Shell Scripting',
        'wireshark': 'Wireshark',
        'selenium': 'Selenium',
        'junit': 'JUnit',
        'excel': 'Excel',
        'microsoft excel': 'Excel',
        'wireframing': 'Wireframing',
        'ui': 'UI/UX',
        'ux': 'UI/UX',
        'user experience': 'UI/UX',
        'user interface': 'UI/UX',
        'seo': 'SEO',
        'search engine optimization': 'SEO',
        'ga': 'Google Analytics',
        'google analytics': 'Google Analytics',
        'ads': 'Google Ads',
        'google ads': 'Google Ads'
    } 

    normalized_skills = []
    for skill in skills:
        cleaned_skill = skill.strip().lower()
        mapped_skill = skill_mapping.get(cleaned_skill, skill)
        if mapped_skill == skill and ' ' not in mapped_skill:
            mapped_skill = mapped_skill.capitalize()
        normalized_skills.append(mapped_skill)

    return normalized_skills

def recommend_for_api(skills, top_n=3):
    user_skills = normalize_skills(skills)
    user_text = " ".join(user_skills)
    user_embedding = model.encode(user_text)

    similarities = []
    for job_role, job_info in job_embeddings.items():
        sim = cosine_similarity(
            user_embedding.reshape(1, -1),
            job_info['embedding'].reshape(1, -1)
        )[0][0]
        similarities.append((job_role, sim, job_info['skills']))

    similarities.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for job_role, score, job_skills in similarities[:top_n]:
        normalized_job_skills = set(normalize_skills(job_skills))
        normalized_user_skills = set(user_skills)
        skill_gaps = normalized_job_skills - normalized_user_skills
        gap_resources = {}
        for gap in skill_gaps:
            for job in jobs_data:
                if gap in job['essential_skills'] and gap in job['learning_resources']:
                    gap_resources[gap] = job['learning_resources'][gap]
                    break

        recommendations.append({
            'role': job_role,
            'match_score': round(score * 100, 2),
            'skill_gaps': list(skill_gaps),
            'gap_resources': gap_resources
        })

    return recommendations

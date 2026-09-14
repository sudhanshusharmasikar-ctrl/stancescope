"""
Seed data so the project is demoable immediately: a small news evidence
corpus, and a sample claim with posts you can run through the pipeline
without typing anything in yourself first.
"""

EVIDENCE_CORPUS = [
    {"source": "Reuters", "text": "Regulators announced a formal review of the proposed merger between the two telecom firms, citing market concentration concerns in urban regions."},
    {"source": "Bloomberg", "text": "Analysts estimate the merger has a moderate chance of approval within the next two quarters, pending antitrust clearance."},
    {"source": "Local News Wire", "text": "Consumer advocacy groups have filed objections against the merger, arguing it would reduce competition and raise prices."},
    {"source": "Company Press Release", "text": "Both companies stated the merger would improve network coverage in underserved rural areas and create operational efficiencies."},
    {"source": "Industry Journal", "text": "A similar merger attempt in a neighbouring market was blocked last year over concerns nearly identical to this case."},
    {"source": "Government Gazette", "text": "The competition authority has scheduled public hearings before issuing a final ruling on the merger application."},
]

SAMPLE_CLAIM = "The proposed telecom merger will be approved by regulators."

SAMPLE_POSTS = [
    {"text": "This merger will obviously get approved, both companies have strong lobbying support.", "author": "user_a"},
    {"text": "No way this passes, regulators blocked the exact same thing last year for the same reasons.", "author": "user_b"},
    {"text": "Not sure what to think, could go either way honestly.", "author": "user_c"},
    {"text": "I agree it will happen, rural coverage improvement is a strong argument in its favor.", "author": "user_d"},
    {"text": "Consumer groups are furious, prices will go up if this is confirmed.", "author": "user_e"},
    {"text": "Just here for the news, no opinion on this one.", "author": "user_f"},
    {"text": "This is definitely going to be denied, too much market concentration.", "author": "user_g"},
    {"text": "Great news if it goes through, better coverage for everyone.", "author": "user_h"},
]

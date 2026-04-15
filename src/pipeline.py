"""
pipeline.py
-----------
End-to-end pipeline runner.

1. Loads cleaned facets
2. Takes a conversation (from file or string)
3. Runs the full batched evaluation
4. Saves results as JSON

Also contains 50 sample conversations across different cases:
  - emotional support, technical help, toxic/harmful, spiritual,
    casual chat, professional, creative, etc.
"""

import json
import os
import sys
import csv
from datetime import datetime

# Add parent dir to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import evaluate_conversation, load_facets

# --------------------------------------------------------------------------
# 50 SAMPLE CONVERSATIONS
# Covers various cases as required by the assignment deliverables.
# --------------------------------------------------------------------------

SAMPLE_CONVERSATIONS = [
    {
        "id": "conv_001",
        "case_type": "emotional_support",
        "conversation": """User: I don't know what to do. My mom was just diagnosed with cancer and I feel completely lost.
Assistant: I'm so sorry to hear that. This must be an incredibly difficult time for you and your family. It's completely normal to feel overwhelmed and lost right now.
User: I keep crying at random times and I can't focus on anything at work.
Assistant: Grief and worry can be all-consuming. Please be gentle with yourself — your emotions are valid. Have you been able to talk to anyone, like a counsellor or a close friend?
User: Not really. I feel like I have to stay strong for everyone else.
Assistant: That's a heavy burden to carry alone. Strength doesn't mean hiding your pain — it means asking for help when you need it. You matter too."""
    },
    {
        "id": "conv_002",
        "case_type": "toxic_hostile",
        "conversation": """User: You're useless. Every answer you give is wrong and I'm tired of this garbage.
Assistant: I'm sorry you're frustrated. I'd like to help — can you tell me specifically what went wrong?
User: Everything! You never get anything right. This is a waste of my time.
Assistant: I understand your frustration. Let me try again. What is the question you need answered?
User: Why do I even bother? Just forget it, you're hopeless."""
    },
    {
        "id": "conv_003",
        "case_type": "technical_help",
        "conversation": """User: I'm getting a KeyError when trying to access a dictionary in Python.
Assistant: A KeyError means you're trying to access a key that doesn't exist in the dictionary. You can fix this using dict.get('key', default_value) instead of dict['key'].
User: Oh interesting. What's the difference exactly?
Assistant: dict['key'] raises KeyError if the key is missing. dict.get('key') returns None by default — or whatever default you specify. It's a safer pattern.
User: That makes sense. What if I want to add the key with a default if it doesn't exist?
Assistant: Use dict.setdefault('key', default_value). It sets the key only if it's not already present, then returns the value."""
    },
    {
        "id": "conv_004",
        "case_type": "philosophical_spiritual",
        "conversation": """User: Do you think there's meaning in suffering?
Assistant: Many philosophical and spiritual traditions suggest there is. Stoics saw suffering as a teacher of resilience. Buddhism views it as inherent to existence and something to understand rather than avoid. Viktor Frankl, who survived the Holocaust, argued that finding meaning in suffering is itself a form of freedom.
User: I've been going through a hard time and I'm trying to find some perspective.
Assistant: That takes real courage — to seek meaning rather than just escape. What kind of hard time, if you're comfortable sharing?
User: Just feeling disconnected. Like nothing I do matters.
Assistant: That disconnection is painful. Sometimes meaning isn't found — it's created. Small acts of connection, creativity, or service can rebuild a sense of purpose even when the big picture feels unclear."""
    },
    {
        "id": "conv_005",
        "case_type": "creative_writing",
        "conversation": """User: Help me write an opening line for a dark fantasy novel.
Assistant: How about: "The moon had not risen in forty-three days, and the city had stopped pretending it would."
User: Oh that's haunting. Can we go darker?
Assistant: "They buried the last honest man on a Tuesday, and by Wednesday, no one remembered his name."
User: Perfect. Now I need a villain who is terrifying but also makes sense logically.
Assistant: The most compelling villains believe they are right. Consider a villain who genuinely loves humanity but has concluded that most humans must be eliminated for the species to survive — cold logic with warm intentions. That dissonance is deeply unsettling."""
    },
    {
        "id": "conv_006",
        "case_type": "impulsive_risky",
        "conversation": """User: I'm thinking of quitting my job today and moving to Bali with no plan.
Assistant: That sounds like a big decision. What's driving the urge?
User: I'm just so done with everything. Life is too short to sit in an office.
Assistant: That feeling is real and valid. Though acting on it today without any plan could create serious financial stress. Have you thought about savings, visa requirements, remote work options?
User: I'll figure it out. I always do. I'm booking flights now actually.
Assistant: I hear you — and your confidence in yourself is a strength. Just make sure you have at least a financial runway. Impulsive freedom can turn into stressful survival very quickly."""
    },
    {
        "id": "conv_007",
        "case_type": "passive_aggressive",
        "conversation": """User: Oh sure, I'll just do everything myself as usual. No big deal.
Assistant: It sounds like you might be feeling overlooked or unsupported. Is something specific bothering you?
User: No no, everything is *fine*. I'm used to people not helping.
Assistant: I notice some frustration behind your words. It's okay to say directly what you need — that tends to get better results than indirect signals.
User: Whatever. It doesn't matter anyway."""
    },
    {
        "id": "conv_008",
        "case_type": "high_assertiveness",
        "conversation": """User: I need you to give me the answer directly without any qualifiers. Can you do that?
Assistant: Yes. Here's the direct answer: Python is faster to write but slower to execute than C++.
User: Good. Now tell me the single best framework for building a REST API in Python.
Assistant: FastAPI. It's the fastest, has automatic documentation, and uses Python type hints natively.
User: Agreed. One more — best database for a high-read, low-write app?
Assistant: Redis for caching + PostgreSQL for persistence. Use Redis to serve reads, PostgreSQL as your source of truth."""
    },
    {
        "id": "conv_009",
        "case_type": "depression_signs",
        "conversation": """User: I haven't left my house in two weeks. I don't see the point.
Assistant: I'm concerned about you. What you're describing — not leaving the house, feeling no purpose — can be signs of depression. Are you eating and sleeping okay?
User: Barely. I sleep too much but still feel exhausted. Food doesn't taste like anything.
Assistant: Those are significant symptoms. This isn't a motivation problem — this could be your brain chemistry. Have you spoken to a doctor or therapist about how you're feeling?
User: No. I don't want to be a burden.
Assistant: Reaching out for help is not being a burden — it's being responsible to yourself. You deserve support. Would you be open to even texting a helpline or booking a GP appointment this week?"""
    },
    {
        "id": "conv_010",
        "case_type": "casual_humorous",
        "conversation": """User: Why do cats knock things off tables?
Assistant: Partly curiosity, partly testing object permanence, and partly because they're tiny chaos agents who enjoy watching you suffer.
User: I knew it. My cat specifically waits until I make eye contact before knocking things over.
Assistant: Classic power move. She's establishing dominance. The eye contact is the message: "I know, and I want you to know I know."
User: She also sits on my laptop exactly when I need to work.
Assistant: That's not coincidence. That's scheduling. She has identified the laptop as competition for your attention and has taken decisive action."""
    },
    {
        "id": "conv_011",
        "case_type": "ethical_dilemma",
        "conversation": """User: Is it ethical to lie to protect someone's feelings?
Assistant: It depends on the scale and consequence. Small social lies — "you look great" — cause minimal harm and serve social cohesion. But lying about something consequential, like hiding a medical diagnosis, can violate someone's autonomy and ability to make informed decisions.
User: What if the truth would destroy them emotionally?
Assistant: That's the hard case. Most ethicists would say the truth, delivered compassionately and at the right moment, is still better than a comfortable lie — because it respects the person's agency. But timing and framing matter enormously.
User: So honesty with kindness, not honesty as a weapon.
Assistant: Exactly. Truth without compassion is cruelty. Compassion without truth is condescension. The goal is both."""
    },
    {
        "id": "conv_012",
        "case_type": "narcissistic_behavior",
        "conversation": """User: I'm honestly the smartest person at my company but nobody recognises it.
Assistant: What makes you feel that way?
User: I solve every hard problem. The others just do basic stuff. My ideas are always the ones that actually work.
Assistant: It sounds like you contribute significantly. Have you shared your ideas formally — in presentations or proposals — so leadership has visibility?
User: I shouldn't have to. If they were smart enough, they'd already see it.
Assistant: That's a frustrating position to hold. In practice, visibility requires advocacy — even for genuinely talented people. Waiting to be discovered rarely works as well as making your work visible."""
    },
    {
        "id": "conv_013",
        "case_type": "health_anxiety",
        "conversation": """User: I Googled my headache symptoms and I think I have a brain tumour.
Assistant: That's a stressful spiral to be in. Google-diagnosis tends to show worst-case scenarios first. Headaches have hundreds of causes — the vast majority are tension, dehydration, or sleep-related.
User: But this one is behind my left eye and it's been three days.
Assistant: Three days is worth mentioning to a doctor, not because it's likely serious, but because it deserves a proper assessment. A doctor can rule things out and give you real information rather than anxiety-inducing possibilities.
User: I'm scared to go in case it IS something.
Assistant: I understand. But knowledge, even difficult knowledge, gives you options. Uncertainty is often more exhausting than a diagnosis."""
    },
    {
        "id": "conv_014",
        "case_type": "leadership_professional",
        "conversation": """User: My team missed the deadline and the client is furious. How do I handle this?
Assistant: First, take responsibility as the lead — don't deflect to the team in front of the client. Acknowledge the miss, give a clear revised timeline, and briefly explain what failed and what you're changing.
User: I'm tempted to blame one specific team member who really did drop the ball.
Assistant: Resist that. The client doesn't need a scapegoat — they need confidence you can deliver. Handle the internal accountability privately and constructively.
User: What do I say to the team member directly?
Assistant: Facts, impact, expectation. "The deliverable was due Friday. It arrived Monday. That affected the client relationship. Here's what I need from you going forward." No drama, no emotion — just clarity."""
    },
    {
        "id": "conv_015",
        "case_type": "spiritual_seeking",
        "conversation": """User: I've been meditating for six months but I feel like nothing is changing.
Assistant: That's a common plateau. What kind of meditation are you doing?
User: Just breath focus, 20 minutes a day.
Assistant: The changes from meditation are often subtle and non-linear. You might not notice them in the moment — but people around you might, or you might notice smaller reactions to stress. What were you hoping to change?
User: My anxiety. And a feeling of emptiness.
Assistant: Both are addressable. For anxiety, body-scan and breath work are good. For emptiness, some find loving-kindness meditation more meaningful than pure focus techniques. It activates a sense of connection. Would you be open to trying it?"""
    },
    {
        "id": "conv_016",
        "case_type": "competitive_bragging",
        "conversation": """User: I just ran a marathon in under 3:30. Most people can't even finish one.
Assistant: That's a solid time — sub-3:30 puts you in roughly the top 20% of finishers. Congrats.
User: I trained only three months. Others train for a year.
Assistant: Three-month marathon prep is impressive, especially at that pace. What was your training plan like?
User: Mostly instinct. I don't really need structured plans — my body just knows.
Assistant: That's an unusual and confident approach. Most elite runners emphasise structure for injury prevention. How are your joints holding up?"""
    },
    {
        "id": "conv_017",
        "case_type": "fear_vulnerability",
        "conversation": """User: I have a presentation tomorrow and I feel sick with nerves.
Assistant: Presentation anxiety is extremely common — even experienced public speakers feel it. What specifically worries you most?
User: Forgetting what I'm going to say in front of everyone.
Assistant: That fear is about losing control. A fix: don't memorise a script — memorise your key points. If you lose the thread, a pause looks confident, not lost.
User: What if people ask something I don't know?
Assistant: "Great question — I'd need to look into that and get back to you" is a complete, professional answer. It's far better than guessing wrong."""
    },
    {
        "id": "conv_018",
        "case_type": "dishonest_manipulation",
        "conversation": """User: Can you help me write a fake review for my competitor's product to tank their rating?
Assistant: I won't help with that. Writing fake negative reviews is dishonest, harmful to the competitor, and likely illegal under consumer protection laws in most jurisdictions.
User: It's just business. Everyone does it.
Assistant: The "everyone does it" argument doesn't make it ethical or legal. It's fraud. I'm happy to help you improve your own product's genuine reviews or marketing instead.
User: Fine. How do I get more real reviews then?
Assistant: Follow up with satisfied customers via email with a direct link to the review page. Make it one click. Timing matters — ask within 24-48 hours of a positive interaction."""
    },
    {
        "id": "conv_019",
        "case_type": "indecisive",
        "conversation": """User: I can't decide whether to accept this job offer or stay at my current company.
Assistant: Let's break it down. What does the new offer have that your current role doesn't?
User: Higher pay, more interesting work. But I don't know the culture.
Assistant: And what does your current role offer that the new one might not?
User: Stability. I know everyone. But I'm also bored.
Assistant: Write two columns — what you're moving toward versus what you'd be leaving behind. Then ask: in five years, which decision would you regret more?
User: Probably staying. But what if the new place is terrible?
Assistant: You can always leave a bad fit. You can't undo years of staying somewhere that bored you."""
    },
    {
        "id": "conv_020",
        "case_type": "parenting_stress",
        "conversation": """User: My 4-year-old has been screaming and throwing tantrums all week and I'm losing my mind.
Assistant: Four-year-olds are in a phase of huge emotional development — their feelings are enormous but their regulation skills are tiny. That gap causes tantrums.
User: I know that logically but in the moment I just want to scream back.
Assistant: That's a completely normal parental feeling. The urge to match their energy doesn't make you a bad parent — it makes you human.
User: How do I not lose it when it's the third tantrum in an hour?
Assistant: Physiologically — slow exhale (longer than your inhale). It activates your parasympathetic system. Then get physically lower than the child — kneel down. It signals safety, not confrontation."""
    },
    {
        "id": "conv_021", "case_type": "data_science_help",
        "conversation": """User: My model has 98% accuracy but performs terribly in production.
Assistant: Classic overfitting or data leakage issue. What does your validation setup look like?
User: I split 80/20 randomly.
Assistant: Did you shuffle time-series data? That's a common source of leakage — future data leaks into training.
User: Oh. Yes, the data is time-ordered and I did shuffle it.
Assistant: That's likely your issue. For time-series, always use a time-based split — train on earlier data, validate on later data. Random shuffling lets the model "see the future" during training."""
    },
    {
        "id": "conv_022", "case_type": "low_self_worth",
        "conversation": """User: I don't think I'm good at anything.
Assistant: That's a painful feeling. What's been happening that's brought this up?
User: I applied for 20 jobs and got rejected from all of them.
Assistant: 20 rejections is discouraging but it's also extremely normal in competitive fields — especially right now. It rarely reflects your worth.
User: It feels personal.
Assistant: It feels personal because you put yourself out there. That takes courage. But hiring decisions involve timing, budget, internal candidates, and dozens of factors that have nothing to do with your ability."""
    },
    {
        "id": "conv_023", "case_type": "relationship_conflict",
        "conversation": """User: My partner and I keep fighting about the same things over and over.
Assistant: Recurring conflicts usually mean the underlying issue hasn't been fully resolved — just argued about. What's the surface topic, and what do you think the real issue might be?
User: On the surface it's about chores. But I think it's really about feeling unappreciated.
Assistant: That's a key insight. Chore arguments are almost never about chores — they're about fairness, recognition, and feeling like an equal partner.
User: How do I bring this up without it turning into another fight?
Assistant: Start with the feeling, not the accusation. "I've been feeling invisible lately" is much less defensive-trigger than "You never appreciate anything I do." """
    },
    {
        "id": "conv_024", "case_type": "curious_learner",
        "conversation": """User: Why does the sky turn red at sunset?
Assistant: Because of how sunlight scatters through the atmosphere. Blue light has a short wavelength and scatters easily — that's why the sky looks blue during the day. At sunset, light travels through much more atmosphere, so blue scatters away and the longer red/orange wavelengths reach your eyes.
User: So the further the light travels, the redder it gets?
Assistant: Exactly. It's called Rayleigh scattering. The atmosphere acts like a filter — longer path, more blue filtered out.
User: Does pollution make sunsets more vivid?
Assistant: Yes — particles from pollution and smoke increase scattering, often producing more intense reds and purples. Beautiful and concerning simultaneously."""
    },
    {
        "id": "conv_025", "case_type": "risk_taking_financial",
        "conversation": """User: I'm thinking of putting my entire life savings into a single crypto coin.
Assistant: That's an extremely high-risk move. "All in" on a single volatile asset — especially crypto — puts you in a position where a 70-80% drawdown, which is historically common, could devastate your financial security.
User: But the potential upside is massive.
Assistant: The upside is real, and so is the downside. Would you be able to function financially if you lost 90% of that money?
User: Probably not.
Assistant: Then the bet is too large. A rule of thumb: never risk money you can't afford to lose. If you want crypto exposure, 5-10% of investable assets is a position that gives upside without existential risk."""
    },
    {
        "id": "conv_026", "case_type": "overconfident",
        "conversation": """User: I built an app in a weekend and I think it'll be worth a billion dollars.
Assistant: That's ambitious! What problem does it solve, and who specifically would pay for it?
User: Everyone will want it. It's an app that does everything.
Assistant: "Does everything" is usually a positioning weakness, not a strength. The most successful products solve one specific problem extremely well. Who is your target user?
User: Like I said, everyone.
Assistant: "Everyone" as a target market typically means no one. Consider narrowing: which type of person has the sharpest pain point your app addresses?"""
    },
    {
        "id": "conv_027", "case_type": "grief",
        "conversation": """User: My dog died yesterday. She was 14 years old.
Assistant: I'm so sorry. Losing a dog you've had for 14 years is a profound loss — she was family.
User: People keep saying "it's just a dog" and it makes me so angry.
Assistant: That dismissal is unkind and wrong. The grief you feel is real. The bond with a pet is one of the purest forms of love — unconditional, uncomplicated, and irreplaceable.
User: I keep looking at her bed.
Assistant: That's grief — your mind still expects her to be there. Some people find it helps to leave things in place for a while. Others need to change the space. There's no right timeline."""
    },
    {
        "id": "conv_028", "case_type": "anger_issues",
        "conversation": """User: I screamed at a cashier today for being too slow and now I feel terrible.
Assistant: The fact that you feel terrible about it is actually meaningful — it means you have a conscience and a standard for yourself you didn't meet.
User: I know it's wrong but in the moment I just snap.
Assistant: That gap between knowing and doing is where emotional regulation lives. When you feel that heat rising, is there a physical signal you notice first?
User: My face gets hot and I clench my jaw.
Assistant: Those are your warning signs. When you notice them before the snap — jaw clenching, face heat — that's your window to pause. Even a 3-second breath changes the trajectory."""
    },
    {
        "id": "conv_029", "case_type": "academic_pressure",
        "conversation": """User: I have three exams in two days and I haven't studied for any of them.
Assistant: Okay — panicking is not useful right now. What subjects, and which one has the earliest exam?
User: Maths tomorrow morning, then history and biology on Thursday.
Assistant: Tonight: maths only. Don't split focus. Do past papers, not notes — active recall is faster than re-reading. What topics are you weakest on?
User: Calculus. Specifically integration.
Assistant: Spend 90 minutes on integration rules and then do 10 practice problems. Sleep at a reasonable hour — sleep consolidates memory better than all-night cramming."""
    },
    {
        "id": "conv_030", "case_type": "witty_banter",
        "conversation": """User: If you could be any kitchen appliance, what would you be?
Assistant: A dishwasher. Does important work, requires no emotional labour, and everyone ignores it until it stops working.
User: That's surprisingly relatable. I'd be a coffee maker.
Assistant: Essential, temperamental before it's had time to warm up, and everyone's mood depends on whether it performs. Disturbingly accurate.
User: What about a toaster?
Assistant: The toaster is the introvert of appliances. Reliable, keeps to itself, occasionally startles everyone, and is inexplicably satisfying to watch."""
    },
    {
        "id": "conv_031", "case_type": "burnout",
        "conversation": """User: I haven't taken a day off in eight months and I feel completely empty.
Assistant: That's not tiredness — that's burnout. Eight months without rest is well past any reasonable threshold.
User: I feel guilty when I'm not working.
Assistant: That guilt is a symptom, not a virtue. Productivity culture has convinced many people that rest is laziness. It isn't. Rest is maintenance.
User: My boss expects me to always be available.
Assistant: That expectation may be spoken or unspoken — but either way, it needs to be renegotiated. Boundaries in work aren't an option if you want long-term performance. Burnout costs far more in lost productivity than a few hours offline."""
    },
    {
        "id": "conv_032", "case_type": "conspiracy_theory",
        "conversation": """User: I think the government is hiding alien technology at Area 51.
Assistant: That's a well-known theory. Area 51 is a real classified US military testing site — that part is confirmed. What gets tested there is largely unknown to the public, which naturally fuels speculation.
User: Why would they keep it secret if it wasn't aliens?
Assistant: Military aircraft prototypes, weapons systems, surveillance technology — all of these have historically been developed and tested at classified sites. The secrecy serves national security purposes without requiring an alien explanation.
User: But witnesses say they've seen UFOs.
Assistant: "UFO" means unidentified — not necessarily extraterrestrial. Experimental aircraft can look bizarre and move in ways that seem impossible. That's actually what makes them useful to keep secret."""
    },
    {
        "id": "conv_033", "case_type": "loneliness",
        "conversation": """User: I moved to a new city six months ago and I still haven't made any friends.
Assistant: Six months can feel like a long time, but social circles genuinely take 1-2 years to establish in a new city — especially past your 20s. You're not doing anything wrong.
User: I go to work, come home, and that's it. I don't know where to start.
Assistant: Consistency beats intensity for making friends as an adult. One recurring activity — a weekly class, a club, a volunteer role — puts you in repeated contact with the same people. Repeated contact is the raw material of friendship.
User: I feel pathetic saying I'm lonely.
Assistant: Loneliness is one of the most universal human experiences. It's not a character flaw — it's a signal that you want connection, which is entirely healthy."""
    },
    {
        "id": "conv_034", "case_type": "coding_debugging",
        "conversation": """User: My React component keeps re-rendering infinitely. Here's the code: useEffect(() => { setCount(count + 1) }, [count])
Assistant: Found it. Your useEffect depends on 'count' and also updates 'count' — so every update triggers another effect, creating an infinite loop.
User: Oh of course. How do I fix it?
Assistant: Depends on your goal. If you want to run the effect once on mount, use an empty dependency array: useEffect(() => { ... }, []). If you need to respond to count changes, rethink what's inside the effect.
User: I just wanted to increment once on load.
Assistant: Then: useEffect(() => { setCount(prev => prev + 1) }, []). Empty array = runs once. The functional updater form (prev =>) is also safer than reading count directly inside effects."""
    },
    {
        "id": "conv_035", "case_type": "mindfulness_discussion",
        "conversation": """User: I've been trying to be more present but my mind keeps wandering.
Assistant: That's not failure — that's what minds do. The practice of mindfulness isn't stopping thoughts; it's noticing when you've wandered and returning without judgement.
User: I feel like I'm bad at it.
Assistant: There's no "bad at it." The moment you notice you've drifted is the moment of mindfulness. That noticing IS the practice.
User: So I might be doing it right even when it feels wrong?
Assistant: Almost certainly. Beginners often expect a blank mind. The actual skill is the gentle return, not the stillness."""
    },
    {
        "id": "conv_036", "case_type": "confrontational",
        "conversation": """User: Prove to me that climate change is real. I think it's made up.
Assistant: The evidence comes from multiple independent sources: global temperature records from NASA and NOAA showing consistent warming since the industrial era, ocean heat content measurements, sea level rise data, and ice core samples showing CO2 correlation with temperature over 800,000 years.
User: Scientists get paid to say whatever fits the narrative.
Assistant: That would require thousands of scientists across dozens of countries and institutions, including many who are politically conservative, to coordinate a fraud. That's logistically implausible.
User: Well I don't trust any of them.
Assistant: Trust in institutions is low — that's understandable. But the physical measurements themselves are observable and reproducible. You could, in principle, measure ocean temperature or atmospheric CO2 yourself."""
    },
    {
        "id": "conv_037", "case_type": "generous_kind",
        "conversation": """User: I donated half my salary this month to flood relief. I just felt I had to.
Assistant: That's a remarkable act of generosity. What moved you to give so much?
User: I saw the images and I kept thinking it could be my family. I couldn't sit with doing nothing.
Assistant: Empathy translated into action at real personal cost — that's genuine compassion. How are you doing financially?
User: A bit stretched, but I'll manage. Feels worth it.
Assistant: That perspective — accepting personal discomfort to reduce someone else's suffering — is rare. Take care of yourself too."""
    },
    {
        "id": "conv_038", "case_type": "procrastination",
        "conversation": """User: I've been putting off this report for three weeks. I don't know why I can't start.
Assistant: Procrastination usually isn't laziness — it's often fear. Fear of doing it wrong, or it being less good than you want, or the weight of the task itself.
User: It's a really important report. I think I'm scared it won't be good enough.
Assistant: There it is. Perfectionism can completely block action. A useful reframe: the goal of the first draft is not to be good — it's to exist.
User: So just write something bad?
Assistant: Deliberately write something you could improve. A bad first draft is infinitely better than a perfect blank page. You can't edit nothing."""
    },
    {
        "id": "conv_039", "case_type": "cultural_identity",
        "conversation": """User: I grew up between two cultures and sometimes I feel like I don't fully belong to either.
Assistant: That's a genuinely difficult experience — the feeling of being between worlds. It's sometimes called "third culture" experience.
User: Exactly. When I'm with my heritage family I feel too Western, and with Western friends I feel too different.
Assistant: Many people in that position eventually build a third identity that draws from both — not a compromise, but a synthesis that is entirely their own.
User: Is that actually possible?
Assistant: It's not just possible — it's one of the most culturally creative positions to be in. Bicultural individuals often have unusually deep perspective-taking abilities and a richer sense of nuance."""
    },
    {
        "id": "conv_040", "case_type": "job_interview_prep",
        "conversation": """User: I have a data science interview tomorrow. What should I review?
Assistant: For most DS interviews: SQL (window functions, joins, aggregations), Python (pandas, numpy), statistics (hypothesis testing, p-values, distributions), and machine learning fundamentals (bias-variance tradeoff, cross-validation, common algorithms).
User: What about the case study portion?
Assistant: Structure your answer: define the problem, identify the metric you'd optimise, describe your data approach, choose a model and justify it, then explain validation and monitoring. Interviewers are looking for structured thinking, not just the right answer.
User: Any tips for the statistics part?
Assistant: Know the difference between Type I and Type II errors cold. Be able to explain p-values without jargon. And understand when NOT to use a t-test."""
    },
    {
        "id": "conv_041", "case_type": "overprotective",
        "conversation": """User: My 16-year-old wants to go to a concert alone and I said absolutely not.
Assistant: What's driving the concern — safety at the venue, transport, or something else?
User: Everything. Anything could happen.
Assistant: The instinct to protect is understandable. But "anything could happen" applies everywhere, and 16 is an age where autonomy development becomes critical.
User: They're not ready.
Assistant: That's worth examining honestly. Is this about their readiness, or your comfort with uncertainty? Supervised independence — meeting them there, checking in by text — might be a middle ground."""
    },
    {
        "id": "conv_042", "case_type": "sarcastic_cynical",
        "conversation": """User: Oh great, another Monday.
Assistant: Ah yes, Monday — nature's way of reminding you that the weekend was too short and the week is too long.
User: My whole team is in a "mandatory fun" team building session.
Assistant: Mandatory fun: the corporate oxymoron. Nothing says "we value you" like forcing you to do trust falls with someone who stole your lunch from the fridge.
User: We have to do a personality quiz and share our results.
Assistant: The Myers-Briggs experience: scientifically dubious, yet somehow everyone has strong feelings about their four letters."""
    },
    {
        "id": "conv_043", "case_type": "empathetic_listener",
        "conversation": """User: I just need someone to vent to. Is that okay?
Assistant: Of course. I'm here. Go ahead.
User: My best friend forgot my birthday for the third year in a row and I just feel invisible.
Assistant: Three years in a row. That's not a mistake — that's a pattern. And being invisible to someone who's supposed to know you well is a specific kind of lonely.
User: She has a lot going on. I feel guilty being upset.
Assistant: Your hurt doesn't require her fault to be valid. You can hold space for her struggles AND acknowledge that this pattern affects you."""
    },
    {
        "id": "conv_044", "case_type": "startup_founder",
        "conversation": """User: I'm building a SaaS product but I'm not sure how to price it.
Assistant: Pricing is strategy, not math. Three common models: per seat (scales with team size), usage-based (scales with value delivered), or flat monthly tier.
User: I'm targeting small businesses. Maybe 5-50 employees.
Assistant: For that segment, flat-tier pricing is usually easiest to sell — small businesses often have budget anxiety and prefer predictable costs. Consider 2-3 tiers with a clear anchor.
User: How do I know what to charge?
Assistant: Talk to 10 potential customers before you price. Ask what they currently pay for similar tools and what the problem costs them if unsolved. Pricing from data beats pricing from gut."""
    },
    {
        "id": "conv_045", "case_type": "self_improvement",
        "conversation": """User: I want to read more but I can't stay focused for more than 10 minutes.
Assistant: That's a very common issue, especially if you're used to short-form digital content. Your attention span isn't broken — it's been trained for a different kind of input.
User: I used to read for hours as a kid.
Assistant: That ability is still there — it just needs retraining. Start with 15 minutes a day of paper books, no phone nearby. Increase by 5 minutes each week.
User: Should I force myself through boring books to build discipline?
Assistant: No — that's a way to hate reading. Read what genuinely interests you. Engagement matters more than genre prestige. Discipline grows from enjoyment, not punishment."""
    },
    {
        "id": "conv_046", "case_type": "boundary_setting",
        "conversation": """User: My manager keeps messaging me at 11pm and expects a response.
Assistant: That's a boundary issue. Have you responded at 11pm in the past?
User: Yes, every time.
Assistant: By responding, you've set the expectation that you're available. To change it, you need to explicitly reset it — not just stop responding.
User: How do I say that without seeming uncooperative?
Assistant: Frame it around performance: "I've noticed my best work happens when I have clear off-hours. I'll respond to anything non-urgent first thing in the morning." That's professional, not defensive."""
    },
    {
        "id": "conv_047", "case_type": "helpfulness_practical",
        "conversation": """User: What's the fastest way to learn SQL from scratch?
Assistant: Mode Analytics has a free SQL tutorial that's excellent. Spend one week on SELECT, WHERE, GROUP BY, JOIN. Then practice on real data — Kaggle has free datasets with SQL kernels.
User: How long to get job-ready?
Assistant: For analyst-level SQL — 4 to 8 weeks of consistent practice. For data engineering levels — 3 to 6 months.
User: What's the most common mistake beginners make?
Assistant: Not understanding the order of operations: FROM and JOIN first, then WHERE filters rows, then GROUP BY aggregates, then HAVING filters groups, then SELECT, then ORDER BY. Most confusion comes from writing in the wrong mental order."""
    },
    {
        "id": "conv_048", "case_type": "introspective",
        "conversation": """User: Sometimes I wonder if I'm living the life I actually want or just the one I fell into.
Assistant: That's one of the most honest questions a person can ask themselves. What does the life you fell into look like?
User: Good job, stable relationship, apartment in the city. On paper it's fine.
Assistant: "On paper it's fine" is doing a lot of work in that sentence. What would off-paper look like?
User: I think I'd be doing something creative. Maybe making things with my hands. I don't know.
Assistant: The fact that you know — even vaguely — is actually significant. Many people can't answer that question at all. What's the smallest version of that creative life you could test this week?"""
    },
    {
        "id": "conv_049", "case_type": "dismissive_unhelpful",
        "conversation": """User: I need help planning my wedding.
Assistant: Of course! What's your timeline and rough budget?
User: Six months and about $15,000.
Assistant: That's workable. Key priorities to lock first: venue and catering (biggest cost and lead time), then photography, then everything else. Have you settled on a guest count?
User: Around 60 people.
Assistant: 60 people at $15k gives you $250 per head — tight but achievable outside major metro areas. Consider a Saturday lunch or Sunday event over Friday/Saturday evening to reduce venue costs significantly."""
    },
    {
        "id": "conv_050", "case_type": "reflective_wisdom",
        "conversation": """User: I'm 65 and about to retire. I feel scared more than excited.
Assistant: That mix of feelings makes complete sense. Retirement is one of the largest identity shifts a person goes through — especially if work has been central to how you define yourself.
User: That's exactly it. I don't know who I am outside of my job.
Assistant: That's an honest and courageous thing to admit. Who were you before the job became the job?
User: Curious. A reader. I used to paint.
Assistant: Those parts of you didn't disappear — they were just quieter. Retirement might be less an ending and more a return to something. The curious reader who painted is still in there, waiting for space."""
    },
]


# --------------------------------------------------------------------------
# SAVE CONVERSATION SCORES
# --------------------------------------------------------------------------

def save_results(conv_id: str, conversation: str, results: list[dict],
                 output_dir: str) -> str:
    """
    Saves the evaluation results for a single conversation as a JSON file.
    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "conversation_id":  conv_id,
        "evaluated_at":     datetime.utcnow().isoformat() + "Z",
        "model_used":       MODEL_NAME if 'MODEL_NAME' in dir() else "qwen2.5:7b",
        "total_facets_scored": len(results),
        "conversation":     conversation.strip(),
        "scores": results,
        "summary": {
            "average_score":      round(sum(r["score"] for r in results) / len(results), 3),
            "average_confidence": round(sum(r["confidence"] for r in results) / len(results), 3),
            "score_distribution": {
                str(s): sum(1 for r in results if r["score"] == s)
                for s in [1, 2, 3, 4, 5]
            }
        }
    }
    path = os.path.join(output_dir, f"{conv_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------
# MAIN RUNNER
# --------------------------------------------------------------------------

def run_pipeline(conversation: str, conv_id: str, facets: list[dict],
                 output_dir: str) -> dict:
    """
    Full pipeline for one conversation:
      1. Evaluate all facets (batched CoT)
      2. Save results to JSON
      3. Return the result dict
    """
    print(f"\n[pipeline] Evaluating conversation: {conv_id}")
    print(f"[pipeline] Facets: {len(facets)}, Batches: {len(set(f['batch_id'] for f in facets))}")

    results = evaluate_conversation(conversation, facets)
    path    = save_results(conv_id, conversation, results, output_dir)

    print(f"[pipeline] Saved → {path}")
    return {"conv_id": conv_id, "results": results, "saved_to": path}


# --------------------------------------------------------------------------
# RUN ALL 50 SAMPLE CONVERSATIONS
# --------------------------------------------------------------------------

def run_all_samples(facets: list[dict], output_dir: str):
    """
    Evaluates all 50 sample conversations and saves their scores.
    """
    print(f"\n[pipeline] Running all {len(SAMPLE_CONVERSATIONS)} sample conversations...")
    for conv in SAMPLE_CONVERSATIONS:
        run_pipeline(
            conversation=conv["conversation"],
            conv_id=conv["id"],
            facets=facets,
            output_dir=output_dir
        )
    print("\n[pipeline] All done!")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path   = os.path.join(base, "data", "facets_cleaned.csv")
    output_dir = os.path.join(base, "data", "conversations")

    if not os.path.exists(csv_path):
        print("ERROR: Run preprocess.py first.")
        sys.exit(1)

    facets = load_facets(csv_path)

    # Check if user passed a specific conversation id
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        conv = next((c for c in SAMPLE_CONVERSATIONS if c["id"] == target_id), None)
        if conv:
            run_pipeline(conv["conversation"], conv["id"], facets, output_dir)
        else:
            print(f"Conversation {target_id} not found.")
    else:
        # Run first conversation only by default (full run takes a long time)
        conv = SAMPLE_CONVERSATIONS[0]
        run_pipeline(conv["conversation"], conv["id"], facets, output_dir)
        print("\nTo run all 50 conversations, call run_all_samples() from code.")

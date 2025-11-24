from __future__ import annotations

import json
import re
from typing import Optional, TYPE_CHECKING

from mcp_agent.core.context import Context as AppContext
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.logging.logger import get_logger

from models.job import Job
from models.insights import SkillGap, ActionStep, Insights

if TYPE_CHECKING:
    from main import Response, UserContext

logger = get_logger(__name__)


async def synthesize_insights(
    response: "Response",
    user_context: "UserContext",
    app_ctx: Optional[AppContext] = None
) -> None:
    """Synthesize personalized career insights using AugmentedLLM and research from insight tools."""
    logger.info(f"synthesize_insights: starting for query: {response.query}")

    # Get research from response.insights (populated by insight tools)
    role_research = response.insights.tavily_role_research or {}
    market_trends = response.insights.tavily_market_trends or {}
    learning_resources = response.insights.tavily_learning_resources or {}

    # Build context from jobs
    job_skills = set()
    job_titles = []
    salary_ranges = []

    for job in response.jobs[:10]:  # Top 10 jobs
        job_titles.append(job.title)
        job_skills.update(job.tags)
        if job.salary_min and job.salary_max:
            salary_ranges.append(f"${job.salary_min:,}-${job.salary_max:,}")

    # Format research results
    role_context = "\n".join([
        f"- {r.get('title', '')}: {r.get('content', '')[:200]}"
        for r in role_research.get("results", [])[:3]
    ])

    market_context = "\n".join([
        f"- {r.get('title', '')}: {r.get('content', '')[:200]}"
        for r in market_trends.get("results", [])[:3]
    ])

    learning_context = "\n".join([
        f"- {r.get('title', '')} ({r.get('url', '')})"
        for r in learning_resources.get("results", [])[:5]
    ])

    prompt = f"""You are a career advisor analyzing job market data and providing personalized recommendations.

USER'S TARGET ROLE: {response.query}
USER'S CURRENT SKILLS: {', '.join(user_context.user_skills) if user_context.user_skills else 'Not specified'}

JOB MARKET DATA:
- Job titles found: {', '.join(job_titles[:5])}
- Skills in demand: {', '.join(list(job_skills)[:15])}
- Salary ranges: {', '.join(salary_ranges[:5]) if salary_ranges else 'Varies'}

ROLE RESEARCH:
{role_context}

MARKET TRENDS:
{market_context}

LEARNING RESOURCES:
{learning_context}

Based on this analysis, provide:

1. A brief personalized message (2-3 sentences) summarizing the user's position and key next steps.

2. Skill gaps analysis - compare user's skills to market demands. For each gap, specify:
   - skill name
   - priority: "essential" (must-have for roles) or "supplemental" (nice-to-have)
   - brief description of why it matters

3. Action plan - 3-5 specific, actionable steps with:
   - action description
   - timeline (e.g., "3 days", "1 week", "2 weeks")
   - impact: "high", "medium", or "low"
   - url if there's a relevant learning resource

Format your response as JSON:
{{
    "message": "Your personalized message here",
    "skill_gaps": [
        {{"skill": "name", "priority": "essential", "description": "why it matters"}}
    ],
    "action_plan": [
        {{"action": "what to do", "timeline": "how long", "impact": "high", "url": "optional url or null"}}
    ]
}}

Return ONLY the JSON, no other text."""

    try:
        async with Agent(
            name="career_insights_generator",
            instruction="You are a career advisor that provides actionable insights based on job market data.",
            server_names=[],
        ) as agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            result = await llm.generate_str(message=prompt)

        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            insights_data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")

        message = insights_data.get("message", "Unable to generate personalized message.")

        skill_gaps = [
            SkillGap(
                skill=gap.get("skill", "Unknown"),
                priority=gap.get("priority", "supplemental"),
                description=gap.get("description", "")
            )
            for gap in insights_data.get("skill_gaps", [])
        ]

        action_plan = [
            ActionStep(
                action=step.get("action", ""),
                timeline=step.get("timeline", "TBD"),
                impact=step.get("impact", "medium"),
                url=step.get("url")
            )
            for step in insights_data.get("action_plan", [])
        ]

        insights = Insights(skill_gaps=skill_gaps, action_plan=action_plan)
        logger.info(f"synthesize_insights: generated {len(skill_gaps)} skill gaps, {len(action_plan)} action steps")

        # Update response with comprehensive insight
        response.insights.comprehensive = {
            "message": message,
            **insights.to_dict()
        }
        response.metadata.insights_generated = True

        if app_ctx:
            app_ctx.logger.info("Insights generated", data={
                "skill_gaps": len(insights.skill_gaps),
                "action_steps": len(insights.action_plan)
            })

    except Exception as e:
        logger.error(f"synthesize_insights: error: {e}")
        # Set default comprehensive insight on error
        response.insights.comprehensive = {
            "message": f"I found {len(response.jobs)} relevant jobs for {response.query}. Review the listings to identify skill requirements.",
            "skill_gaps": [],
            "action_plan": []
        }

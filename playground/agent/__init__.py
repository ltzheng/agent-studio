# import argparse

# from playground.agent.base_agent import Agent
# from playground.agent.direct_agent import DirectAgent
# from playground.llm.lm_config import construct_llm_config


# def construct_agent(args: argparse.Namespace) -> Agent:
#     llm_config = construct_llm_config(args)

#     agent: Agent
#     if args.agent_type == "direct":
#         agent = DirectAgent(
#             lm_config=llm_config,
#         )
#     else:
#         raise NotImplementedError(f"agent type {args.agent_type} not implemented")
#     return agent

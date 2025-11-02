# Lazy imports to avoid loading torch/langchain at startup
# Phase 3 agents are loaded on-demand to prevent DLL initialization issues
def _lazy_import(module_name):
    """Lazy import helper to avoid circular dependencies and heavy imports"""
    import importlib
    module_parts = module_name.split('.')
    module = importlib.import_module('.' + module_parts[-1], package='.'.join(module_parts[:-1]))
    return module

def __getattr__(name):
    """Lazy loading of agent modules only when accessed"""
    if name in ["AgentState", "InvestDebateState", "RiskDebateState"]:
        from .utils.agent_states import AgentState, InvestDebateState, RiskDebateState
        return locals()[name]
    elif name == "FinancialSituationMemory":
        from .utils.memory import FinancialSituationMemory
        return FinancialSituationMemory
    elif name == "create_msg_delete":
        from .utils.agent_utils import create_msg_delete
        return create_msg_delete
    elif name.startswith("create_"):
        # Lazy load specific agent creators
        if name.endswith("_analyst"):
            from . import analysts
            agent_type = name.replace("create_", "").replace("_analyst", "")
            if agent_type == "fundamentals":
                from .analysts.fundamentals_analyst import create_fundamentals_analyst
                return create_fundamentals_analyst
            elif agent_type == "market":
                from .analysts.market_analyst import create_market_analyst
                return create_market_analyst
            elif agent_type == "news":
                from .analysts.news_analyst import create_news_analyst
                return create_news_analyst
            elif agent_type == "social_media":
                from .analysts.social_media_analyst import create_social_media_analyst
                return create_social_media_analyst
        elif name.endswith("_researcher"):
            agent_type = name.replace("create_", "").replace("_researcher", "")
            if agent_type == "bear":
                from .researchers.bear_researcher import create_bear_researcher
                return create_bear_researcher
            elif agent_type == "bull":
                from .researchers.bull_researcher import create_bull_researcher
                return create_bull_researcher
        elif name.endswith("_debator"):
            agent_type = name.replace("create_", "").replace("_debator", "")
            if agent_type == "risky":
                from .risk_mgmt.aggresive_debator import create_risky_debator
                return create_risky_debator
            elif agent_type == "safe":
                from .risk_mgmt.conservative_debator import create_safe_debator
                return create_safe_debator
            elif agent_type == "neutral":
                from .risk_mgmt.neutral_debator import create_neutral_debator
                return create_neutral_debator
        elif name == "create_research_manager":
            from .managers.research_manager import create_research_manager
            return create_research_manager
        elif name == "create_risk_manager":
            from .managers.risk_manager import create_risk_manager
            return create_risk_manager
        elif name == "create_trader":
            from .trader.trader import create_trader
            return create_trader
    raise AttributeError(f"module 'tradingagents.agents' has no attribute '{name}'")

__all__ = [
    "FinancialSituationMemory",
    "AgentState",
    "create_msg_delete",
    "InvestDebateState",
    "RiskDebateState",
    "create_bear_researcher",
    "create_bull_researcher",
    "create_research_manager",
    "create_fundamentals_analyst",
    "create_market_analyst",
    "create_neutral_debator",
    "create_news_analyst",
    "create_risky_debator",
    "create_risk_manager",
    "create_safe_debator",
    "create_social_media_analyst",
    "create_trader",
]

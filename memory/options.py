from typing import List, Optional


class OptionModule:
    """Maps abstract options to concrete action candidates."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def choose_option(self, observation: str) -> Optional[str]:
        if not self.enabled:
            return None

        lowered = (observation or "").lower()
        if "locked" in lowered:
            return "inspect"
        if "inventory" in lowered or "carrying" in lowered:
            return "interact"
        if "cannot" in lowered or "you can't" in lowered:
            return "explore"
        return "explore"

    def option_actions(self, option: Optional[str]) -> List[str]:
        if option == "explore":
            return ["look", "north", "south", "east", "west", "help"]
        if option == "inspect":
            return ["look", "examine door", "examine room", "inventory"]
        if option == "interact":
            return ["inventory", "use", "open", "take all", "drop"]
        return []

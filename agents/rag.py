import argparse
import os

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

from agents.llm import LLMAgent
from tales.agent import register

class RAGAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weaviate_url = os.environ.get("WEAVIATE_URL")
        weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
        
        if not weaviate_url or not weaviate_api_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in the environment.")
            
        # Use the OpenAI API Key from the LLM model configuration
        openai_key = self.model.key if self.model.key else os.environ.get("OPENAI_API_KEY", "")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            headers={"X-OpenAI-Api-Key": openai_key}
        )
        
        self.rag_top_k = kwargs.get("rag_top_k", 3)
        
        # Weaviate collection names must start with a capital letter and contain only alphanumeric chars and underscores.
        safe_uid = "".join(c if c.isalnum() else "_" for c in self.uid)
        self.collection_name = "TalesMemoryV2_" + safe_uid
        
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=[
                    Property(name="observation", data_type=DataType.TEXT),
                    Property(name="action", data_type=DataType.TEXT),
                    Property(name="feedback", data_type=DataType.TEXT),
                    Property(name="reward", data_type=DataType.NUMBER),
                ],
            )
        self.collection = self.client.collections.get(self.collection_name)
        
        self.last_obs = None
        self.last_action = None

    @property
    def uid(self):
        return super().uid + f"_rag{getattr(self, 'rag_top_k', 3)}"

    @property
    def params(self):
        p = super().params
        p["agent_type"] = "rag"
        p["rag_top_k"] = getattr(self, "rag_top_k", 3)
        return p

    def reset(self, obs, info, env_name):
        super().reset(obs, info, env_name)
        self.last_obs = None
        self.last_action = None

    def build_messages(self, observation):
        messages = super().build_messages(observation)
        
        # Retrieve context from Weaviate
        if self.rag_top_k > 0:
            response = self.collection.query.near_text(
                query=observation,
                limit=self.rag_top_k
            )
            
            if response.objects:
                context_str = "Recall from similar past situations:\n"
                for obj in response.objects:
                    props = obj.properties
                    context_str += f"- When you saw '{props.get('observation', '')}', you did '{props.get('action', '')}' and the result was '{props.get('feedback', '')}' (Reward: {props.get('reward', 0)}).\n"
                
                # Append context to the last user message
                messages[-1]["content"] += f"\n\n[Memory Context]\n{context_str}"
                
        return messages

    def act(self, obs, reward, done, infos):
        feedback = infos.get("feedback", "")
        # Store previous memory
        if self.last_obs is not None and self.last_action is not None:
            self.collection.data.insert({
                "observation": self.last_obs,
                "action": self.last_action,
                "feedback": feedback,
                "reward": float(reward),
            })
            
        action, stats = super().act(obs, reward, done, infos)
        
        self.last_obs = obs
        self.last_action = action
        
        return action, stats


def build_argparser(parser=None):
    from agents.llm import build_argparser as llm_build_argparser
    parser = llm_build_argparser(parser)
    group = parser.add_argument_group("RAGAgent settings")
    group.add_argument(
        "--rag-top-k",
        type=int,
        default=3,
        help="Number of retrieved memories. Default: %(default)s",
    )
    return parser

register(
    name="rag",
    desc="Retrieval-Augmented Generation agent using Weaviate.",
    klass=RAGAgent,
    add_arguments=build_argparser,
)

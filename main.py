# import basic library
import torch
import numpy as np 

from scripts.topology import topology_mininet
from scripts.env_mininet import MininetGraphEnv
from scripts.agent import TopologyAgent
from scripts.train import train_actor_critic


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create Topology
    topo = topology_mininet()

    # Create environment
    env = MininetGraphEnv(topo)

    # Create agent
    agent = TopologyAgent(env, epsilon = 1, tau = 0.001)

    
    print('Start Training Process--------------------------------------')
    
    # try:
    # Train the agent
    trained_agent = train_actor_critic(
        env, 
        agent,
        actor_lr = 1e-6, 
        critic_lr = 1e-6, 
        gamma = 0.99
    )

    # Optional: Save the trained model
    torch.save(trained_agent.actor.state_dict(), 'trained_actor.pth')
    torch.save(trained_agent.critic.state_dict(), 'trained_critic.pth')

    print("Training completed successfully!")

    # except:
    #     print('Errors occuured')
    #     # env.topology.net.stop()

    #     # # # If processes are not shutting down, you can manually kill them.
    #     # # for node in env.topology.net.values():
    #     # #     node.terminate()  # Terminate all processes

    #     # # If still needed, you can call Mininet's cleanup function:
    #     # env.topology.net.cleanup()

if __name__ == "__main__":
    main()


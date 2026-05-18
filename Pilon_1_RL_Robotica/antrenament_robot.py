import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
# Importăm clientul care ne conectează direct la CoppeliaSim prin ZeroMQ
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class CoppeliaRobotEnv(gym.Env):
    """Mediu personalizat pentru robot în CoppeliaSim conform standardului OpenAI/Gymnasium"""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CoppeliaRobotEnv, self).__init__()
        
        # 1. Conectarea la simulator
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        # Preluăm „ID-urile” unice ale obiectelor din scenă ca să le putem controla prin cod
        self.robot_handle = self.sim.getObject('/Pioneer_p3dx')
        self.left_motor = self.sim.getObject('/Pioneer_p3dx/leftMotor')
        self.right_motor = self.sim.getObject('/Pioneer_p3dx/rightMotor')
        self.target_handle = self.sim.getObject('/Target')

        # 2. Definirea Spațiului de Acțiuni (Ce poate face robotul?)
        # Avem 3 acțiuni discrete: 0 = Mergi în față, 1 = Rotește stânga, 2 = Rotește dreapta
        self.action_space = spaces.Discrete(3)

        # 3. Definirea Spațiului de Stări / Observații (Ce știe/vede robotul?)
        # Vector format din: [Distanța până la țintă, Unghiul față de țintă]
        # Low și High reprezintă valorile minime și maxime posibile ale acestor date
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi]), 
            high=np.array([20.0, np.pi]), 
            dtype=np.float32
        )

    def _get_obs(self):
        """Funcție internă care calculează starea curentă a robotului din simulator"""
        # Luăm poziția robotului și a țintei pe axele X, Y, Z
        robot_pos = self.sim.getObjectPosition(self.robot_handle, self.sim.handle_world)
        target_pos = self.sim.getObjectPosition(self.target_handle, self.sim.handle_world)
        
        # Calculăm distanța Euclidiană clasică pe planul X-Y
        distanta = np.sqrt((robot_pos[0] - target_pos[0])**2 + (robot_pos[1] - target_pos[1])**2)
        
        # Unghiul simplificat (pentru început, punem 0, dar aici se va calcula orientarea robotului)
        unghi = 0.0 
        
        return np.array([distanta, unghi], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Această funcție repornește simularea la începutul fiecărui episod nou de antrenament"""
        super().reset(seed=seed)
        
        # Oprim și repornim simularea fizică din CoppeliaSim
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01)
            
        self.sim.startSimulation()
        
        # Punem ținta (bila roșie) într-o poziție aleatorie la fiecare restart ca robotul să nu învețe mecanic drumul
        rand_x = np.random.uniform(-2.0, 2.0)
        rand_y = np.random.uniform(-2.0, 2.0)
        self.sim.setObjectPosition(self.target_handle, self.sim.handle_world, [rand_x, rand_y, 0.2])
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Aici se întâmplă magia: Robotul execută o acțiune și vedem ce se întâmplă în simulator"""
        
        # Executăm acțiunea aleasă de algoritmul IA
        if action == 0:   # Mergi în față
            self.sim.setJointTargetVelocity(self.left_motor, 2.0)
            self.sim.setJointTargetVelocity(self.right_motor, 2.0)
        elif action == 1: # Rotește stânga
            self.sim.setJointTargetVelocity(self.left_motor, -0.5)
            self.sim.setJointTargetVelocity(self.right_motor, 2.0)
        elif action == 2: # Rotește dreapta
            self.sim.setJointTargetVelocity(self.left_motor, 2.0)
            self.sim.setJointTargetVelocity(self.right_motor, -0.5)

        # Lăsăm simulatorul să ruleze un mic pas de timp fizic
        self.client.step() 
        
        # Luăm noua stare a robotului după ce s-a mișcat
        observation = self._get_obs()
        distanta_curenta = observation[0]

        # ---- CONSTRUIREA FUNCȚIEI DE RECOMPENSĂ (Reward Shaping) ----
        # Îi dăm puncte cu cât distanța e mai mică de țintă (invers proporțional)
        reward = 1.0 / (distanta_curenta + 0.01)
        
        terminated = False
        # Dacă a ajuns foarte aproape de bilă (sub 25 cm), înseamnă că a câștigat episodul!
        if distanta_curenta < 0.25:
            reward += 100.0 # Recompensă uriașă!
            terminated = True
            
        # Limită de timp sau ieșire din decor (opțional)
        truncated = False 
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Oprește simularea definitiv când închidem programul"""
        self.sim.stopSimulation()

# =====================================================================
# BARA DE LANSARE ȘI ANTRENAMENT PROPRIU-ZISĂ
# =====================================================================
if __name__ == "__main__":
    from stable_baselines3 import PPO

    print("Inițializare mediu robot...")
    env = CoppeliaRobotEnv()

    # Folosim algoritmul PPO (unul dintre cei mai stabili și moderni algoritmi de RL)
    print("Se creează modelul de IA bazat pe algoritmul PPO...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    # Robotul va interacționa cu mediul din simulator timp de 10.000 de pași pentru a învăța
    print("Începe antrenamentul robotului în CoppeliaSim. Privește ecranul simulatorului!")
    model.learn(total_timesteps=10000)

    # Salvarea creierului robotului după ce a învățat
    model.save("creier_robot_mobil")
    print("Antrenament finalizat cu succes! Modelul a fost salvat.")
    
    env.close()
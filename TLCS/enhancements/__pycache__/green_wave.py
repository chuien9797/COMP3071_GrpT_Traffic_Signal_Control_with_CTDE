import traci
import time

INTERSECTIONS = ["tls_1", "tls_2", "tls_3"]
ROAD_LENGTHS = [200, 250]  # Distance in meters
AVG_SPEED = 13.89  # Speed in m/s (~50 km/h)

def calculate_offsets():
    return [round(dist / AVG_SPEED) for dist in ROAD_LENGTHS]

def set_green_wave(offsets):
    for i in range(1, len(INTERSECTIONS)):
        upstream = INTERSECTIONS[i - 1]
        downstream = INTERSECTIONS[i]
        delay = offsets[i - 1]

        current_phase = traci.trafficlight.getPhase(upstream)
        traci.trafficlight.setPhase(downstream, current_phase)
        time.sleep(delay)

def monitor_emissions():
    total_co2 = sum(traci.vehicle.getCO2Emission(veh) for veh in traci.vehicle.getIDList())
    return total_co2

sumoCmd = ["sumo-gui", "-c", "intersection\sumo_config.sumocfg"]
traci.start(sumoCmd)

offsets = calculate_offsets()

try:
    step = 0
    while step < 300:
        traci.simulationStep()
        
        if step % 10 == 0:
            set_green_wave(offsets)
        
        if step % 50 == 0:
            print(f"Step {step}: CO2 = {monitor_emissions():.2f} mg")

        step += 1
finally:
    traci.close()

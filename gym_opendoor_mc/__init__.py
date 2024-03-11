from gym.envs.registration import register

# register(
#     id='Ingress-v0',
#     entry_point='gym_ingress_mc.envs:IngressEnv',
# )
# register(
#     id='Ingress-v1',
#     entry_point='gym_ingress_mc.envs:IngressEnvSimple',
# )
register(
    id='OpenDoor-v4',
    entry_point='gym_opendoor_mc.envs:OpenDoorEnv',
)

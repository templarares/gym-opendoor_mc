import mc_rtc_rl
from mc_rtc_rl import Configuration, ConfigurationException
"""
convert fsm name to numerical values; initial is 0,righthandtocar is 1, etc...; then normalize it to the obserrvation space
"""
def StateNumber(name):
    stateNumber_=-20
    if (name=="Initial"):
        stateNumber_=0
    elif (name=="OpenDoorRLFSM::Standing"):
        stateNumber_=1
    elif (name=="OpenDoorRLFSM::RH2HandleApproach"):
        stateNumber_=2
    elif (name=="OpenDoorRLFSM::RH2HandleAbove"):
        stateNumber_=3
    elif (name=="OpenDoorRLFSM::RH2HandleDown"):
        stateNumber_=4
    elif (name=="OpenDoorRLFSM::RH2HandlePush"):
        stateNumber_=5
    elif (name=="OpenDoorRLFSM::RHDisengage"):
        stateNumber_=6
    elif (name=="OpenDoorRLFSM::Rewind"):
        stateNumber_=7    
    elif (name=="OpenDoorRLFSM::LHReach"):
        stateNumber_=8
        """change these state numbers and male sure they are not duplicate when it comes to full ingress"""
    elif (name=="OpenDoorRLFSM::LHPush"):
        stateNumber_=9
    elif (name=="OpenDoorRLFSM::Teleport"):
        stateNumber_=10
    elif (name=="OpenDoorRLFSM::LHPushAgain"):
        stateNumber_=11
    elif (name=="OpenDoorRLFSM::PutLeftFoot::LiftFoot"):
        stateNumber_=12
    elif (name=="OpenDoorRLFSM::PutLeftFoot::MoveFoot"):
        stateNumber_=13
    elif (name=="OpenDoorRLFSM::PutLeftFoot::PutFoot"
    or name=="OpenDoorRLFSM::PutLeftFoot"):
        stateNumber_=14
    elif (name=="OpenDoorRLFSM::PutRightFoot"):
        stateNumber_=15
    elif (name=="OpenDoorRLFSM::NudgeUp"):
        stateNumber_=16
    elif (name=="OpenDoorRLFSM::ScootRight"):
        stateNumber_=17
    elif (name=="OpenDoorRLFSM::SitOnLeft"):
        stateNumber_=18
    # elif (name=="OpenDoorRLFSM::NedgeUp"):
    #     stateNumber_=19
    # elif (name=="OpenDoorRLFSM::CoMToRightFoot"):
    #     stateNumber_=5

    # elif (name=="OpenDoorRLFSM::PutLeftHand"):
    #     stateNumber_=7
    # elif (name=="OpenDoorRLFSM::PutLeftFoot"):
    #     stateNumber_=8
    # elif (name=="OpenDoorRLFSM::PutRightFoot"):
    #     stateNumber_=9
    # elif (name=="OpenDoorRLFSM::ScootRight"):
    #     stateNumber_=10
    # elif (name=="OpenDoorRLFSM::ScootRightFoot"):
    #     stateNumber_=11
    # elif (name=="OpenDoorRLFSM::ScootAdjustHand"):
    #     stateNumber_=12
    # elif (name=="OpenDoorRLFSM::ScootBody"):
    #     stateNumber_=13
    # elif (name=="OpenDoorRLFSM::ScootLeftFoot"):
    #     stateNumber_=14
    # elif (name=="OpenDoorRLFSM::SitPrep"):
    #     stateNumber_=15
    #normalize it to the range [-2,+2]
    #return stateNumber_*0.1
    import numpy as np
    stateVec=np.zeros((20,))
    stateVec[stateNumber_]=1
    return stateVec

def EditTimeout(config,timeout,eval=0.01, speed=0.01):
    OR=config.array("OR")
    EVAL=mc_rtc_rl.Configuration()
    EVAL.add("eval",float(eval))
    OR.push(EVAL)
    ANDconfig=mc_rtc_rl.Configuration()
    AND=ANDconfig.array("AND")
    TIMEOUT=mc_rtc_rl.Configuration()
    TIMEOUT.add("timeout",float(abs(timeout)))
    AND.push(TIMEOUT)
    SPEED=mc_rtc_rl.Configuration()
    SPEED.add("speed",float(speed))
    AND.push(SPEED)
    OR.push(ANDconfig)
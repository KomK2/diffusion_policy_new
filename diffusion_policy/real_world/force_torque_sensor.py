import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import rospy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger
from leptrino_force_torque.srv import getForceTorque


class FTSensor(mp.Process):
    def __init__(self, 
                 shm_manager: SharedMemoryManager,
                 service_name = 'force_torque_service',
                 obs_data_key = 'ft_data',
                 get_max_k=500,
                 launch_timeout = 3,
                 verbose = True
                 ):
        example = dict()
        example[obs_data_key] = np.zeros((6,), dtype=np.float64)
        example['ft_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=500
        )

        self.ring_buffer = ring_buffer
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.service_name = service_name
        self.obs_data_key = obs_data_key
        self.calibration_coeff = mp.Array('d', [0.0] * 6)
        rospy.wait_for_service(self.service_name)
        self.get_latest_ft_data = rospy.ServiceProxy('force_torque_service', getForceTorque)
        
        
        super().__init__()

    def calibrate_sensor(self):
        calibration_data = []
        previous_data = None
        for i in range(300):
            this_service = self.get_latest_ft_data()
            this_data = [this_service.ft_data.wrench.force.x, 
                        this_service.ft_data.wrench.force.y, 
                        this_service.ft_data.wrench.force.z,
                        this_service.ft_data.wrench.torque.x, 
                        this_service.ft_data.wrench.torque.y, 
                        this_service.ft_data.wrench.torque.z]
            
            # Check if all elements in this_data are zero
            if all(x == 0 for x in this_data):
                # If all elements are zero, use previous non-zero data
                if previous_data is not None:
                    this_data = previous_data
            else:
                # Update previous non-zero data
                previous_data = this_data
            
            calibration_data.append(this_data)
            # time.sleep(0.001)
        calibration_data = np.array(calibration_data, dtype=np.float64)
        with self.calibration_coeff.get_lock():
            self.calibration_coeff[:] = np.mean(calibration_data, axis=0)
        print("Sensor Calibration Done, Coeff:", np.frombuffer(self.calibration_coeff.get_obj()))

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait()
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        try:
            iter_idx = 0
            self.calibrate_sensor()
            previous_raw_data = None
            while not self.stop_event.is_set(): 
                data = self.get_latest_ft_data()
                state = dict()
                raw_ft_data = np.array([data.ft_data.wrench.force.x, 
                                        data.ft_data.wrench.force.y, 
                                        data.ft_data.wrench.force.z,
                                        data.ft_data.wrench.torque.x, 
                                        data.ft_data.wrench.torque.y, 
                                        data.ft_data.wrench.torque.z], dtype=np.float64)
                if all(x == 0 for x in raw_ft_data):
                    # If all elements are zero, use previous non-zero data
                    if previous_raw_data is not None:
                        raw_ft_data = previous_raw_data
                else:
                    # Update previous non-zero data
                    previous_raw_data = raw_ft_data

                with self.calibration_coeff.get_lock():
                    # print(self.calibration_coeff[:])
                    calibrated_data = raw_ft_data - np.frombuffer(self.calibration_coeff.get_obj(), dtype=np.float64)    
                
                calibrated_data = np.round((calibrated_data),6)
                state[self.obs_data_key] = calibrated_data
                state['ft_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx+=1

        finally:
            self.ready_event.set()
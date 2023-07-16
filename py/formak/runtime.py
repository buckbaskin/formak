class ManagedFilter(object):
    def tick(self, output_time, readings=None):
        for sensor_reading in readings:
            self.state = process_model(sensor_reading.time, self.state)
            self.state = sensor_model(sensor_reading, self.state)

        return process_model(self.state)

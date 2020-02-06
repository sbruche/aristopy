import os
import sys
import logging
import aristopy


class Logger:
    def __init__(self, logfile_name='logfile.log', delete_old_logs=True,
                 default_log_handler='stream', local_log_handler={},
                 default_log_level='DEBUG', local_log_level={},
                 write_screen_output_to_logfile=False):
        """
        The Logger class can be used to manage the logging of the energy
        system model instance and the associated component instances.

        :param logfile_name: Name of the file to store the logging.
        :type logfile_name: string
        
        :param delete_old_logs: Delete old log-files before creating a new file?
            Otherwise new logs might be appended to an old log-file.
        :type delete_old_logs: boolean
                      
        :param default_log_handler: Sets a default handler to the loggers. The
            handler options are "file" or "stream":
            |br| * "file" sends logs to a file with the specified "logfile_name"
            |br| * "stream" sends logs to the console (sys.stdout).
        :type default_log_handler: string

        :param local_log_handler: Dictionary to override the default handler
            level for specified instance types ('EnergySystemModel', 'Source',
            'Sink', 'Conversion', 'Storage', 'Bus'). E.g.: ... = {'Bus': 'file'}
        :type local_log_handler: dict

        :param default_log_level: Sets the default threshold for the logger
            level for all derived instance loggers. Logging messages which are
            less severe than level will be ignored. Options for logger levels
            are: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        :type default_log_level: string

        :param local_log_level: Dictionary to override the default log level for
            specified instance types ('EnergySystemModel', 'Source', 'Sink',
            'Conversion', 'Storage', 'Bus'). E.g.: ... = {'Bus': 'WARNING', ...}
        :type local_log_level: dict

        :param write_screen_output_to_logfile: State whether the output that is
            printed on the console should be written to the logfile too.
        :type write_screen_output_to_logfile: boolean
        """

        # Check user input:
        aristopy.check_logger_input(logfile_name, delete_old_logs,
                                    default_log_handler, local_log_handler,
                                    default_log_level, local_log_level,
                                    write_screen_output_to_logfile)

        self.logfile = logfile_name

        # Delete all log files in the current working directory
        if delete_old_logs:
            for file in os.listdir(os.getcwd()):
                if file.endswith('.log'):
                    os.remove(file)

        # Dictionary for assigning the possible names (keys) from the dicts
        # "local_log_handler" and "local_log_level" to the classes of aristopy
        self.instance_types = {'EnergySystemModel': aristopy.EnergySystemModel,
                               'Source': aristopy.Source,
                               'Sink': aristopy.Sink,
                               'Conversion': aristopy.Conversion,
                               'Storage': aristopy.Storage,
                               'Bus': aristopy.Bus}

        # Specify a dictionary with log levels for all instance types
        # (either default or overwritten with "local_log_level")
        self.default_log_level = default_log_level
        self.log_level = {}
        for instance in self.instance_types.keys():
            if instance in local_log_level.keys():
                self.log_level.update(
                    {self.instance_types[instance]: local_log_level[instance]})
            else:
                self.log_level.update(
                    {self.instance_types[instance]: self.default_log_level})

        # Specify a dictionary with log handlers for all instance types
        # (either default or overwritten with "local_log_handler")
        self.default_log_handler = default_log_handler
        self.log_handler = {}
        for instance in self.instance_types.keys():
            if instance in local_log_handler.keys():
                self.log_handler.update(
                    {self.instance_types[instance]: local_log_handler[instance]})
            else:
                self.log_handler.update(
                    {self.instance_types[instance]: self.default_log_handler})

        # ---------------------------------------------------------
        # Redirect the general screen output to the logfile as well
        if write_screen_output_to_logfile:
            sys.stdout = ConsoleLogger(logfile_name=self.logfile)
        # ---------------------------------------------------------

    def _get_log_handler(self, instance):
        # Set logger output to stream to console or write to log-file
        if self.log_handler[type(instance)] == 'file':
            return logging.FileHandler(self.logfile, 'a')  # append
        else:  # self.log_handler[type(instance)] == 'stream':
            return logging.StreamHandler()

    def get_logger(self, instance):
        """
        Generate a new logger instance with default logger level and default
        logger handle. The name of the logger is specified in "instance".
        If an instance of class 'EnergySystemModel', 'Source', 'Sink',
        'Conversion', 'Storage', 'Bus' is passed to the function, local log
        handler and local log levels might be used.

        :param instance: String (name of the logger) or instance of aristopy
            class type 'EnergySystemModel', 'Source', 'Sink', 'Conversion',
            'Storage', 'Bus'.
        :return: New logger instance
        """
        # Define the format of the output:
        # https://www.pylenin.com/blogs/python-logging-guide/
        # https://docs.python.org/3/library/logging.html#logrecord-attributes
        log_format = '%(asctime)s - %(filename)20s (line: %(lineno)4d) ' \
                     '- %(name)-24s - %(levelname)-8s >> %(message)s <<'
        # Specify the format of the output and add the instance handler
        formatter = logging.Formatter(log_format)

        # If instance is not derived from aristopy classes 'EnergySystemModel',
        # 'Source', 'Sink', 'Conversion', 'Storage', 'Bus' --> new logger with
        # default settings is generated.
        if type(instance) not in self.instance_types.values():
            # Name the logger and set logging level and handle to default
            logger = logging.getLogger(instance)
            level = self.default_log_level
            logger.setLevel(level)
            if self.default_log_handler == 'file':
                handler = logging.FileHandler(self.logfile, 'a')
            else:
                handler = logging.StreamHandler()
        else:
            # Get logger name from instance representation __repr__ and the log
            # levels and handles from the dictionary (default or overwritten)
            logger = logging.getLogger(repr(instance))
            level = self.log_level[type(instance)]
            logger.setLevel(level)
            handler = self._get_log_handler(instance)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info('Set logger to level "{}" for instance {}.'.format(
            level, instance))

        return logger


class ConsoleLogger:
    def __init__(self, logfile_name='logfile.log'):
        """
        Class to redirect the output of "stdout" to the console and a logfile.
        See: https://stackoverflow.com/questions/14906764/

        :param logfile_name: Name of file to store the redirected output.
        :type logfile_name: string
        """
        # if os.path.isfile(filename):
        #     os.remove(filename)
        self.terminal = sys.stdout
        self.filename = logfile_name

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as log:
            # Sometimes last sign is a new line operator --> avoid empty lines
            if message[-1] == '\n':
                log.write(message[:-1])
            else:
                log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == '__main__':
    logs = Logger()
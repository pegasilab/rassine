from dataclasses import dataclass
import argparse
from argparse import ArgumentParser, Action, Namespace
from typing import Optional, List, Dict, Set, Sequence, Tuple, Mapping
from pathlib import Path
import os
import sys
from configparser import ConfigParser, SectionProxy
from functools import total_ordering
import logging


class EnrichedAction:
    """
    Stores the extra data enriching an `argparse.Action` when used in `.ParametersParser`

    The enriched action must act as a "store" action, because of the way environment variables / configuration file
    values override each other.

    Attributes:
        action: Action being enriched
        dest: Destination value being stored
        config_var_name: Configuration variable name
        env_var_name: Environment variable name
    """

    def __init__(self, action: Action, env_var_name: Optional[str] = None):
        self.action: Action = action
        self.dest: str = action.dest
        self.config_var_name: Optional[str] = self.extract_config_var_name(action)
        self.env_var_name: Optional[str] = env_var_name

    def __repr__(self):
        return f"ActionExtraData(dest={self.dest}, env_var_name={self.env_var_name}, action={self.action})"

    def __hash__(self):
        return hash(self.dest)

    def __eq__(self, other):
        if isinstance(other, EnrichedAction):
            return self.dest == other.dest
        else:
            return False

    @staticmethod
    def extract_config_var_name(action: Action) -> Optional[str]:
        """
        Returns the first long option string with the ``-`` cut off

        Args:
            action: Action to analyze

        Returns:
            The name of the long option string, used as a configuration file key
        """
        candidates = [s for s in action.option_strings if len(s) >= 1 and s[0] != '-']
        candidates.extend([s[2:] for s in action.option_strings if s.startswith('--')])
        if candidates:
            return candidates[0]
        else:
            return None

    @staticmethod
    def is_store_like_action(action: Action) -> bool:
        """
        Returns whether an action has a "store user value-like" behavior

        We block the argparse types that do not store a user provided value; for other types, we assume that
        the action is valid.

        Args:
            action: Action to analyze

        Returns:
            Whether the action can be driven by a configuration key
        """

        #
        invalid_types = [argparse._StoreConstAction, argparse._AppendAction, argparse._CountAction,
                         argparse._HelpAction, argparse._VersionAction, argparse._SubParsersAction,
                         argparse._ExtendAction, argparse._StoreTrueAction, argparse._StoreFalseAction]

        return not any([isinstance(action, t) for t in invalid_types])


@dataclass
class ConfigSection(object):
    """
    Description of a configuration section to read in a configuration file

    Attributes:
        name: Name of the section header
        strict: Whether the parsing is strict, i.e. all keys in the configuration file must correspond to
                parameters (strict=True), or whether to ignore extra keys (strict=False)
    """
    name: str
    strict: bool = False


class ParametersParser(ArgumentParser):
    """
    Drop-in replacement for `argparse.ArgumentParser` that supports for environment variables and configuration files

    Inspired by `<https://github.com/bw2/ConfigArgParse>`_ but allows for the provision of multiple configuration files
    that override each other parameters.

    Note that relative configuration file paths are resolved according to the current directory when calling the
    `.parse_args` method.

    The constructed parser is single-use.

    Attributes:
        used: Whether the parser has been used already
        enriched_actions: List of `.EnrichedAction` instances, excluding the configuration action
        invalid_configfile_actions: Actions that cannot be used as config. file keys, including the conf. action itself
        config_common_section: Name of the common parameter section in configuration files
        config_sections: Sections to read in the configuration file
        config_parser: First parser called to extract configuration file paths
        config_action: Enriched action that appends configuration file paths (with the action in the config_parser)
    """

    def __init__(self, *args, config_common_section: str = 'common',
                 relaxed_config_sections: Sequence[str] = [], strict_config_sections: Sequence[str] = [], **kwargs):
        """
        Constructs a ParametersParser

        Args:
            config_sections: Sections of the configuration file to use
            strict_config_sections: Sections of the config. file for which all keys should correspond to parameters
        """
        self.used = False
        self.enriched_actions: List[EnrichedAction] = []
        self.invalid_configfile_actions: List[Action] = []
        self.config_common_section: str = config_common_section
        self.config_sections: Sequence[ConfigSection] = [ConfigSection(name, name in strict_config_sections)
                                                         for name in (relaxed_config_sections + strict_config_sections)]
        self.config_parser: ArgumentParser = ArgumentParser()
        self.config_action: Optional[EnrichedAction] = None
        super().__init__(*args, **kwargs)

    def add_config_argument(self, *args, env_var_name: Optional[str] = None, **kwargs) -> Action:
        """
        Adds a parameter that describes a configuration file input

        This method has the same calling convention as `argparse.ArgumentParser.add_argument`.

        This method can only be called once, must have 'action' set to 'append', must have 'type' set to 'pathlib.Path'

        Args:
            env_var_name: Name of the environment variable containing the default configuration file

        Returns:
            The construction configuration file action
        """
        assert self.config_action is None, 'The method add_config_argument must be called at parser construction'

        # Add the configuration action to the configuration parser
        config_action = self.config_parser.add_argument(*args, **kwargs)  # it is the only option of the config. parser
        self.config_action = EnrichedAction(config_action, env_var_name)

        # Add the configuration action to this parser, so that the option shows in the documentation/help
        action: Action = super().add_argument(*args, **kwargs)
        assert isinstance(action, argparse._AppendAction), \
            "For the configuration file option, the action must be 'append'"
        assert action.type == Path, "For configuration file option, the type of must pathlib.Path"
        self.invalid_configfile_actions.append(action)

        return action

    def add_argument(self, *args, env_var_name: Optional[str] = None, **kwargs) -> Action:
        """
        Adds a parameter to the argument parser

        Args:
        env_var_name: Name of the environment variable containing the default configuration file

        Returns:
            The constructed action
        """
        action: Action = super().add_argument(*args, **kwargs)
        if EnrichedAction.is_store_like_action(action):
            enriched_action = EnrichedAction(action, env_var_name)
            self.enriched_actions.append(enriched_action)
        else:
            self.invalid_configfile_actions.append(action)
        return action

    @staticmethod
    def get_env_variable(action: EnrichedAction, env: Mapping[str, str]) -> Optional[str]:
        """
        Returns the value of the given action/parameter if it is present in the environment

        Args:
            action: Enriched action to find the value of
            env: Environment variables

        Returns:
            The parameter value if present, None is not present
        """
        env_var_name = action.env_var_name
        if env_var_name is not None:
            if env_var_name in env:
                return env[env_var_name]
        return None

    @staticmethod
    def explode_config_filenames(filenames: str) -> Sequence[Path]:
        return list([Path(name.strip()) for name in filenames.split(',') if name.strip()])

    def get_config_files(self, args: Sequence[str], env: Mapping[str, str]) -> List[Path]:
        """
        Parses the environment and the command-line arguments to retrieve the list of configuration files to use

        Args:
            args: Command-line arguments
            env: Environment variables

        Returns:
            Path to configuration files
        """
        config_action = self.config_action
        assert config_action is not None, 'A ParametersParser must define exactly one config argument'

        config_files: List[Path] = []

        config_env = self.get_env_variable(config_action, env)

        if config_env is not None:
            new_files: List[Path] = list(self.explode_config_filenames(config_env))
            if new_files:
                logging.info('Using configuration files from the environment: ' + ','.join(map(str, new_files)))
            config_files.extend(new_files)

        config_args, discard = self.config_parser.parse_known_args(args)
        config_cmdline_args = getattr(config_args, config_action.dest)
        if config_cmdline_args:
            new_files = []
            for filenames in config_cmdline_args:
                new_files.extend(self.explode_config_filenames(filenames))
            if new_files:
                logging.info('Using configuration files from the command-line: ' + ','.join(map(str, new_files)))
            config_files.extend(new_files)

        return config_files

    def populate_from_env_variables(self, env: Mapping[str, str]):
        """
        Populate parameter values from environment variables

        Args:
            env: Environment variables
        """
        env_var_names: Dict[str, Action] = dict([(ea.env_var_name, ea.action) for ea in self.enriched_actions
                                                 if ea.env_var_name is not None])
        for name, value in env.items():
            if name in env_var_names:
                action = env_var_names[name]
                logging.info(f'Set {action.dest} to {value} from environment variable {name}')
                action.default = value

    def populate_from_config_section(self, section: SectionProxy, *, strict: bool):
        """
        Populate parameter values from section of configuration file

        Args:
            section: Parsed configuration section
            strict: Whether to parse in strict mode
        """
        invalid_keys: Set[str] = \
            set([name for name in [EnrichedAction.extract_config_var_name(a) for a in self.invalid_configfile_actions]
                 if name is not None])
        config_option_names: Dict[str, Action] = dict([(ea.config_var_name, ea.action) for ea in self.enriched_actions
                                                       if ea.config_var_name is not None])
        for key, value in section.items():
            assert key not in invalid_keys, \
                f"Key {key} is an option but not a valid configuration file key"
            if key in config_option_names:
                logging.info(f'Set {key} to {value} from configuration file')
                action = config_option_names[key]
                action.default = value
            else:
                assert not strict, f"Key {key} is unknown and the section {section.name} is parsed in strict mode"

    def parse_all(self, args: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None,
                  namespace: Optional[Namespace] = None) -> Namespace:
        """
        Parses environment variables, configuration files and command-line arguments

        Args:
            args: Command line arguments
            env: Environment variables
            namespace: Optional namespace to populate

        Returns:
            The populated namespace
        """
        assert not self.used, 'A ParametersParser can only be used once'
        self.used = True
        if args is None:
            args = sys.argv[1:]  # args default to the system args
        if env is None:
            env = os.environ
        config_files = self.get_config_files(args, env)
        self.populate_from_env_variables(env)
        for config_filename in config_files:
            cp = ConfigParser()
            with open(config_filename, 'r') as file:
                cp.read_file(file, str(config_filename))
            try:
                if self.config_common_section in cp.sections():
                    self.populate_from_config_section(cp[self.config_common_section], strict=False)
                for section in self.config_sections:
                    strict = section.strict
                    section_name = section.name
                    if section_name in cp.sections():
                        self.populate_from_config_section(cp[section_name], strict=strict)
            except Exception as exc:
                raise Exception(f"Configuration file {config_filename} could not be properly parsed") from exc
        return super().parse_args(args, namespace=namespace)
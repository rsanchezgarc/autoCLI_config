"""
Integration tests for strict_ambiguity using subprocess to test real conflict detection.

This approach creates actual Python scripts and runs them via subprocess to test
the full CONFIG_PARAM + parser flow in production scenarios.
"""

import subprocess
import tempfile
import os
from pathlib import Path
import pytest
import yaml


def create_test_script(script_path: Path, strict_ambiguity: bool = False):
    """Create a test script that uses ConfigArgumentParser with CONFIG_PARAM."""
    script_content = f'''
import sys
from dataclasses import dataclass
from autoCLI_config import ConfigArgumentParser, CONFIG_PARAM, inject_defaults_from_config

@dataclass
class Config:
    """Test configuration."""
    PARAM_DOCS = {{
        'batch_size': 'Batch size for processing',
        'skip_step': 'Skip processing step'
    }}
    batch_size: int = 32
    skip_step: bool = False

config = Config()

@inject_defaults_from_config(config, update_config_with_args=True)
def main(
    batch_size: int = CONFIG_PARAM(),
    skip_step: bool = CONFIG_PARAM()
):
    """Main function.

    Args:
        batch_size: Batch size
        skip_step: Skip step
    """
    print(f"batch_size={{batch_size}}")
    print(f"skip_step={{skip_step}}")
    print(f"config.batch_size={{config.batch_size}}")
    print(f"config.skip_step={{config.skip_step}}")

if __name__ == "__main__":
    parser = ConfigArgumentParser(config_obj=config, verbose=False, strict_ambiguity={strict_ambiguity})
    parser.add_args_from_function(main)
    args, config_args = parser.parse_args()
    main()
'''
    script_path.write_text(script_content)


class TestStrictAmbiguitySubprocess:
    """Test strict_ambiguity using subprocess for real conflict detection."""

    @staticmethod
    def _get_env_with_local_path():
        """Get environment with PYTHONPATH pointing to local autoCLI_config."""
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent.absolute()
        env['PYTHONPATH'] = str(project_root)
        return env

    def test_strict_ambiguity_false_shows_warning(self, tmp_path):
        """Test that strict_ambiguity=False shows warning and continues."""
        # Create test script
        script_path = tmp_path / "test_script.py"
        create_test_script(script_path, strict_ambiguity=False)

        # Create config file with conflicting value
        yaml_content = {"batch_size": 64}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Run script with CLI arg that conflicts with config
        result = subprocess.run(
            ["python", str(script_path), "--config", str(yaml_file), "--batch_size", "128"],
            capture_output=True,
            text=True,
            env=self._get_env_with_local_path()
        )

        # Should succeed (exit code 0)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Should contain warning about conflict
        assert "Conflict detected" in result.stderr or "UserWarning" in result.stderr

        # Should show CLI value took precedence
        assert "batch_size=128" in result.stdout or "config.batch_size=128" in result.stdout

    def test_strict_ambiguity_true_raises_error(self, tmp_path):
        """Test that strict_ambiguity=True raises error on conflict."""
        # Create test script with strict_ambiguity=True
        script_path = tmp_path / "test_script.py"
        create_test_script(script_path, strict_ambiguity=True)

        # Create config file with conflicting value
        yaml_content = {"batch_size": 64}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Run script with CLI arg that conflicts with config
        result = subprocess.run(
            ["python", str(script_path), "--config", str(yaml_file), "--batch_size", "128"],
            capture_output=True,
            text=True,
            env=self._get_env_with_local_path()
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0, "Script should have failed with strict_ambiguity=True"

        # Should contain error message about conflict
        assert "Conflict" in result.stderr and "ambiguous" in result.stderr

    def test_no_conflict_no_warning(self, tmp_path):
        """Test that matching values don't trigger warnings."""
        # Create test script
        script_path = tmp_path / "test_script.py"
        create_test_script(script_path, strict_ambiguity=False)

        # Create config file with matching value
        yaml_content = {"batch_size": 128}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Run script with CLI arg that matches config
        result = subprocess.run(
            ["python", str(script_path), "--config", str(yaml_file), "--batch_size", "128"],
            capture_output=True,
            text=True,
            env=self._get_env_with_local_path()
        )

        # Should succeed
        assert result.returncode == 0

        # Should NOT contain conflict warning
        assert "Conflict" not in result.stderr

        # Should show correct value
        assert "batch_size=128" in result.stdout

    def test_boolean_flag_conflict(self, tmp_path):
        """Test conflict with boolean flag."""
        # Create test script
        script_path = tmp_path / "test_script.py"
        create_test_script(script_path, strict_ambiguity=False)

        # Create config file with skip_step=False
        yaml_content = {"skip_step": False}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Run script with --skip_step flag (sets to True)
        result = subprocess.run(
            ["python", str(script_path), "--config", str(yaml_file), "--skip_step"],
            capture_output=True,
            text=True,
            env=self._get_env_with_local_path()
        )

        # Should succeed with warning
        assert result.returncode == 0

        # Should show warning
        assert "Conflict detected" in result.stderr or "UserWarning" in result.stderr

        # Should show CLI value (True) took precedence
        assert "skip_step=True" in result.stdout or "config.skip_step=True" in result.stdout

    def test_cli_only_no_warning(self, tmp_path):
        """Test that CLI-only usage doesn't trigger warnings."""
        # Create test script
        script_path = tmp_path / "test_script.py"
        create_test_script(script_path, strict_ambiguity=False)

        # Run script with only CLI arg (no config file)
        result = subprocess.run(
            ["python", str(script_path), "--batch_size", "128"],
            capture_output=True,
            text=True,
            env=self._get_env_with_local_path()
        )

        # Should succeed
        assert result.returncode == 0

        # Should NOT contain warnings
        assert "Conflict" not in result.stderr
        assert "Warning" not in result.stderr

        # Should show CLI value
        assert "batch_size=128" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
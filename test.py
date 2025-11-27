from pathlib import Path
from app.modules.hdl.resource_config import VivadoConfig
from app.modules.hdl.resource_utilization_runner import ResourceUtilizationRunner

config = VivadoConfig(
    settings_script=Path(
        "/mathworks/hub/share/apps/HDLTools/Vivado/2024.1-mw-1/Lin/Vivado/2024.1/settings64.sh"
    ),
    project_root=Path("/tmp/hdlloop_vivado_projects"),
)

runner = ResourceUtilizationRunner(config=config)

report = runner.run_for_files(
    hdl_files=[
        Path("/home/smalviya/HDLLoop/temp/mul_add_pkg.vhd"),  # package
        Path("/home/smalviya/HDLLoop/temp/mul_add.vhd"),      # top
    ],
    top_module="mul_add",
)
print(report)

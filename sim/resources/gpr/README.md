# URDF/XML development:
1. kol run https://cad.onshape.com/documents/bc3a557e2f92bdcea4099f0d/w/09713ac603d461fc1dff6b5d/e/5a4acdbffe6f8ed7c4e34970 --config-path config_example.json # noqa: E501
2. python sim/scripts/create_fixed_torso.py
3. urf2mjcf robot.urdf -o robot.xml
 
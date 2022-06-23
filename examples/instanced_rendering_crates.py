import numpy as np
from pyrr import Matrix44
import struct

import moderngl
import moderngl_window as mglw
from ported._example import Example


class InstancedCrates(Example):
    '''
        This example renders 32x32 crates.
        For each crate the location is [x, y, sin(a * time + b)]
        There are 1024 crates aligned in a grid.
    '''
    title = "Instanced Crates"
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Offscreen render target
        self.offscreen_rgba = self.ctx.texture(self.wnd.buffer_size, 4, dtype="f1")
        self.offscreen_instance_id = self.ctx.texture(self.wnd.buffer_size, 1, dtype="i4")
        self.offscreen_instance_id.filter == (moderngl.NEAREST, moderngl.NEAREST)
        self.offscreen_depth = self.ctx.depth_texture(self.wnd.buffer_size)
        self.offscreen = self.ctx.framebuffer(
            color_attachments=[
                self.offscreen_rgba,
                self.offscreen_instance_id,
            ],
            depth_attachment=self.offscreen_depth,
        )
        self.quad_fs = mglw.geometry.quad_fs()

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_move;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                flat out int v_instance_id;

                void main() {
                    gl_Position = Mvp * vec4(in_position + in_move, 1.0);
                    v_vert = in_position + in_move;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                    // We add one so that 0 only appears in the background.
                    v_instance_id = gl_InstanceID + 1;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform sampler2D Texture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                flat in int v_instance_id;

                out vec4 f_color;
                out int f_instance_id;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                    f_instance_id = v_instance_id;
                }
            ''',
        )

        # This program renders our offscreen texture on a fullscreen quad.
        self.texture_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;
                in vec2 in_texcoord_0;
                out vec2 uv;

                void main() {
                    gl_Position = vec4(in_position, 1.0);
                    uv = in_texcoord_0;
                }
            """,
            fragment_shader="""
                #version 330

                uniform sampler2D texture0;
                in vec2 uv;
                out vec4 f_color;

                void main() {
                    f_color = texture(texture0, uv);
                    if (f_color.a == 0.0) {
                        discard;
                    }
                }
            """
        )
        self.texture_prog["texture0"].value = 0

        # This program retrieves the instance id of the create (if any) under the cursor.
        self.picker_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform isampler2D instance_ids;
                uniform ivec2 texel_pos;

                in vec3 in_position;

                out int instance_id;

                void main() {
                    instance_id = texelFetch(instance_ids, texel_pos, 0).x;
                }
            """,
            varyings=["instance_id"],
        )
        self.picker_prog["instance_ids"].value = 0;
        self.picker_output = self.ctx.buffer(reserve=4) # the output is a single int
        self.picker_vao = mglw.opengl.vao.VAO(mode=moderngl.POINTS)

        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light']

        self.scene = self.load_scene('crate.obj')
        self.texture = self.load_texture_2d('crate.png')

        # Add a new buffer into the VAO wrapper in the scene.
        # This is simply a collection of named buffers that is auto mapped
        # to attributes in the vertex shader with the same name.
        self.instance_data = self.ctx.buffer(reserve=12 * 1024)
        vao_wrapper = self.scene.root_nodes[0].mesh.vao
        vao_wrapper.buffer(self.instance_data, '3f/i', 'in_move')
        # Create the actual vao instance (auto mapping in action)
        self.vao = vao_wrapper.instance(self.prog)

        self.crate_a = np.random.uniform(0.7, 0.8, 32 * 32)
        self.crate_b = np.random.uniform(0.0, 6.3, 32 * 32)
        self.crate_x = (np.tile(np.arange(32), 32) - 16) * 1.5
        self.crate_y = (np.repeat(np.arange(32), 32) - 16) * 1.5
        self.crate_x += np.random.uniform(-0.2, 0.2, 32 * 32)
        self.crate_y += np.random.uniform(-0.2, 0.2, 32 * 32)

    def render(self, time, frame_time):
        angle = time * 0.2
        self.ctx.enable(moderngl.DEPTH_TEST)

        camera_pos = (np.cos(angle) * 5.0, np.sin(angle) * 5.0, 2.0)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
        )

        self.offscreen.use()
        self.offscreen.clear(0.0, 0.0, 0.0)

        self.mvp.write((proj * lookat).astype('f4'))
        self.light.value = camera_pos

        crate_z = np.sin(self.crate_a * time + self.crate_b) * 0.2
        coordinates = np.dstack([self.crate_x, self.crate_y, crate_z])

        self.instance_data.write(coordinates.astype('f4'))
        self.texture.use(location=0)
        self.vao.render(instances=1024)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.screen.use()
        self.ctx.screen.clear(1.0, 1.0, 1.0)
        self.offscreen_rgba.use(location=0)
        self.quad_fs.render(self.texture_prog)

    def mouse_press_event(self, x, y, button):
        super().mouse_press_event(x, y, button)

        # Mouse coordinates start in the upper left, but the pixel positions in our instance
        # texture start in the lower left.
        pos = (
            int(x * self.wnd.pixel_ratio),
            int(self.wnd.buffer_height - (y * self.wnd.pixel_ratio))
        )
        #print(f"screen coords ({x}, {y})")
        #print(f"texture coords {pos}")
        self.picker_prog["texel_pos"].value = pos
        self.offscreen_instance_id.use(location=0)
        self.picker_vao.transform(self.picker_prog, self.picker_output, vertices=1)
        instance_id = struct.unpack("1i", self.picker_output.read())[0]
        if instance_id > 0:
            print(f"you clicked crate instance {instance_id - 1}")
        else:
            print(f"you clicked on the background")


if __name__ == '__main__':
    InstancedCrates.run()

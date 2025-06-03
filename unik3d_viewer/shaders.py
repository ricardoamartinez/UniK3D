# GLSL Shader source strings (copied from live_slam_viewer.py)

VERTEX_SHADER_SOURCE = '''#version 150 core
    in vec3 vertices;
    in vec3 colors; // Expects normalized floats (0.0-1.0)

    out vec3 vertex_colors;
    // Debug outputs
    out float debug_inputDistance_out;
    out float debug_rawDiameter_out;
    out float debug_densityFactor_out;
    out float debug_finalSize_out;

    uniform mat4 projection;
    uniform mat4 view;
    uniform float inputScaleFactor; // Controlled via ImGui
    uniform float pointSizeBoost;   // Controlled via ImGui
    uniform vec2 viewportSize;      // Width, height of viewport in pixels
    uniform float inputFocal;       // Focal length of input camera in pixel units
    uniform float minPointSize;     // Minimum point size in pixels
    uniform float maxPointSize;     // Maximum point size in pixels (if clamp enabled)
    uniform bool enableMaxSizeClamp;// Toggle for max size clamp
    uniform float depthExponent;    // Exponent applied to depth for sizing
    uniform float sizeScaleFactor;  // Tunable scale factor for depth sizing
    uniform bool planarProjectionActive; // NEW UNIFORM

    void main() {
        // Transform to view and clip space
        vec4 viewPos = view * vec4(vertices, 1.0);
        vec4 clipPos = projection * viewPos;
        gl_Position = clipPos;
        vertex_colors = colors;

        // --- Point Sizing based on INPUT Camera Distance (Inverse Square Law) and Density Compensation ---
        float inputDistance = length(vertices);
        inputDistance = max(inputDistance, 0.0001);

        float baseSize = inputFocal * inputScaleFactor;
        float diameter = 2.0 * baseSize * sizeScaleFactor * pointSizeBoost * pow(inputDistance, depthExponent);

        float densityCompensationFactor = 1.0;
        if (!planarProjectionActive) {
            // Apply spherical density compensation only if not in planar projection mode
            vec3 inputRay = normalize(vertices); 
            float cosInputLatitude = sqrt(1.0 - clamp(inputRay.y * inputRay.y, 0.0, 1.0));
            densityCompensationFactor = 1.0 / max(1e-5, cosInputLatitude);
        }

        diameter *= densityCompensationFactor;

        float finalSize = max(diameter, minPointSize);
        if (enableMaxSizeClamp) {
            finalSize = min(finalSize, maxPointSize);
        }
        gl_PointSize = finalSize;

        debug_inputDistance_out = inputDistance;
        debug_rawDiameter_out = 2.0 * baseSize * sizeScaleFactor * pointSizeBoost * pow(inputDistance, depthExponent); 
        debug_densityFactor_out = densityCompensationFactor;
        debug_finalSize_out = finalSize;
    }
'''

FRAGMENT_SHADER_SOURCE = '''#version 150 core
    in vec3 geom_colors; // Input from Geometry Shader
    in vec2 texCoord;    // Input texture coordinate from Geometry Shader
    // Receive debug values
    in float debug_inputDistance_frag;
    in float debug_rawDiameter_frag;
    in float debug_densityFactor_frag;
    in float debug_finalSize_frag;

    out vec4 final_color;

    // Debug uniforms
    uniform bool debug_show_input_distance;
    uniform bool debug_show_raw_diameter;
    uniform bool debug_show_density_factor;
    uniform bool debug_show_final_size;

    uniform int renderMode; // 0=Square, 1=Circle, 2=Gaussian
    uniform float falloffFactor; // For Gaussian
    uniform float saturation;
    uniform float brightness;
    uniform float contrast;
    uniform float sharpness; // Simple contrast boost for sharpening

    // Function to convert RGB to HSV
    vec3 rgb2hsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    // Function to convert HSV to RGB
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    // Simple heatmap function (blue -> green -> red)
    vec3 heatmap(float value) {
        value = clamp(value, 0.0, 1.0); // Ensure value is in [0, 1]
        float r = clamp(mix(0.0, 1.0, value * 2.0), 0.0, 1.0);
        float g = clamp(mix(1.0, 0.0, abs(value - 0.5) * 2.0), 0.0, 1.0);
        float b = clamp(mix(1.0, 0.0, (value - 0.5) * 2.0), 0.0, 1.0);
        return vec3(r, g, b);
    }

    void main() {

        // --- Debug Visualizations ---
        if (debug_show_input_distance) {
            // Map distance (e.g., 0-20) to a heatmap
            float normalized_dist = clamp(debug_inputDistance_frag / 20.0, 0.0, 1.0);
            final_color = vec4(heatmap(normalized_dist), 1.0);
            return; // Skip normal processing
        }
        if (debug_show_raw_diameter) {
             // Map diameter (e.g., 0-50 pixels) to grayscale
             float gray = clamp(debug_rawDiameter_frag / 50.0, 0.0, 1.0);
             final_color = vec4(gray, gray, gray, 1.0);
             return;
        }
        if (debug_show_density_factor) {
             // Map factor (e.g., 1.0 to 5.0+) to heatmap
             float normalized_factor = clamp((debug_densityFactor_frag - 1.0) / 4.0, 0.0, 1.0);
             final_color = vec4(heatmap(normalized_factor), 1.0);
             return;
        }
         if (debug_show_final_size) {
             // Map final size (e.g., 0-50 pixels) to grayscale
             float gray = clamp(debug_finalSize_frag / 50.0, 0.0, 1.0);
             final_color = vec4(gray, gray, gray, 1.0);
             return;
        }
        // --- End Debug Visualizations ---

        // --- Image Processing --- (Only runs if no debug mode active)
        vec3 processed_color = geom_colors;

        // Saturation, Brightness, Contrast (applied in HSV)
        vec3 hsv = rgb2hsv(processed_color);
        hsv.y = clamp(hsv.y * saturation, 0.0, 1.0); // Saturation
        hsv.z = clamp(hsv.z * brightness, 0.0, 1.0); // Brightness
        hsv.z = clamp(0.5 + (hsv.z - 0.5) * contrast, 0.0, 1.0); // Contrast
        processed_color = hsv2rgb(hsv);

        // --- Simple Sharpening via Contrast Boost ---
        vec3 hsv_sharp = rgb2hsv(processed_color);
        hsv_sharp.z = clamp(0.5 + (hsv_sharp.z - 0.5) * sharpness, 0.0, 1.0);
        processed_color = hsv2rgb(hsv_sharp);
        // --- End Sharpening ---


        // --- Shape & Alpha ---
        if (renderMode == 0) { // Square (Opaque)
            final_color = vec4(processed_color, 1.0);
        } else { // Circle or Gaussian
            vec2 coord = texCoord - vec2(0.5);
            float dist_sq = dot(coord, coord);

            if (dist_sq > 0.25) { // Discard if outside circle
                discard;
            }

            if (renderMode == 1) { // Circle (Opaque)
                final_color = vec4(processed_color, 1.0);
            } else { // Gaussian (renderMode == 2)
                // Calculate Gaussian alpha
                float alpha = exp(-4.0 * falloffFactor * dist_sq);
                // Premultiply color by alpha
                vec3 premultipliedRgb = processed_color * alpha;
                final_color = vec4(premultipliedRgb, alpha); // Output premultiplied RGB and alpha
            }
        }
    }
'''

GEOMETRY_SHADER_SOURCE = '''#version 150 core
    layout (points) in;
    layout (triangle_strip, max_vertices = 4) out;

    in vec3 vertex_colors[]; // Receive from vertex shader
    // Receive debug values from vertex shader
    in float debug_inputDistance_out[];
    in float debug_rawDiameter_out[];
    in float debug_densityFactor_out[];
    in float debug_finalSize_out[];

    out vec3 geom_colors;    // Pass color to fragment shader
    out vec2 texCoord;       // Pass texture coordinate to fragment shader
    // Pass debug values to fragment shader
    out float debug_inputDistance_frag;
    out float debug_rawDiameter_frag;
    out float debug_densityFactor_frag;
    out float debug_finalSize_frag;

    uniform vec2 viewportSize; // To convert pixel size to clip space

    void main() {
        vec4 centerPosition = gl_in[0].gl_Position;
        float pointSize = gl_in[0].gl_PointSize; // Get size calculated in vertex shader

        // Calculate half-size in clip space coordinates
        float halfSizeX = pointSize / viewportSize.x;
        float halfSizeY = pointSize / viewportSize.y;

        // Emit 4 vertices for the quad
        gl_Position = centerPosition + vec4(-halfSizeX, -halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(0.0, 0.0); // Bottom-left
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        gl_Position = centerPosition + vec4( halfSizeX, -halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(1.0, 0.0); // Bottom-right
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        gl_Position = centerPosition + vec4(-halfSizeX,  halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(0.0, 1.0); // Top-left
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        gl_Position = centerPosition + vec4( halfSizeX,  halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(1.0, 1.0); // Top-right
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        EndPrimitive();
    }
'''

TEXTURE_VERTEX_SHADER_SOURCE = '''#version 150 core
    in vec2 position;
    in vec2 texCoord_in;
    out vec2 texCoord;

    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        texCoord = texCoord_in;
    }
'''

TEXTURE_FRAGMENT_SHADER_SOURCE = '''#version 150 core
    in vec2 texCoord;
    out vec4 final_color;
    uniform sampler2D fboTexture;
    void main() {
        final_color = texture(fboTexture, texCoord);
    }
'''

DEBUG_VERTEX_SHADER_SOURCE = '''#version 150 core
    in vec3 position;
    in vec3 color;
    out vec3 vertex_colors;

    uniform mat4 projection;
    uniform mat4 view;

    void main() {
        gl_Position = projection * view * vec4(position, 1.0);
        vertex_colors = color;
    }
'''

DEBUG_FRAGMENT_SHADER_SOURCE = '''#version 150 core
    in vec3 vertex_colors;
    out vec4 final_color;

    void main() {
        final_color = vec4(vertex_colors, 1.0);
    }
''' 
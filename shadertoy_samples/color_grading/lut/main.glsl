#iChannel0 "file://../../../images/shang_hai.png"
#iChannel0::WrapMode "Repeat"
#iChannel0::MagFilter "Linear"

#iChannel1 "file://../../../images/LUT_Reddish.webp"

// #iChannel1 "file://../../../images/RGBTable16x1.webp"
#iChannel2::MagFilter "Linear"

#define MAXCOLOR 15.0
#define COLORS 16.0
#define WIDTH 256.0
#define HEIGHT 16.0

vec2 FixUV()
{
    vec2 uv = gl_FragCoord.xy / min(iResolution.x, iResolution.y);

    if(iChannelResolution[0].x > iChannelResolution[0].y)
    {
        uv.y *= iChannelResolution[0].x / iChannelResolution[0].y;
    }
    else
    {
        uv.x *= iChannelResolution[0].x / iChannelResolution[0].y;
    }
    
    return uv;
}

vec3 LookUpTalble(vec3 px)
{
    float cell = px.b * MAXCOLOR;

    float cell_l = floor(cell); // <1>
    float cell_h = ceil(cell);

    float half_px_x = 0.5 / WIDTH;
    float half_px_y = 0.5 / HEIGHT;
    float r_offset = half_px_x + px.r / COLORS * (MAXCOLOR / COLORS);
    float g_offset = half_px_y + px.g * (MAXCOLOR / COLORS);

    vec2 lut_pos_l = vec2(cell_l / COLORS + r_offset, 1. - g_offset); // <2>
    vec2 lut_pos_h = vec2(cell_h / COLORS + r_offset, 1. - g_offset);

    vec3 graded_color_l = texture(iChannel1, lut_pos_l).rgb; // <3>
    vec3 graded_color_h = texture(iChannel1, lut_pos_h).rgb;

    // <4>
    vec3 graded_color = mix(graded_color_l, graded_color_h, fract(cell));
    return graded_color;
}

void main() {
    vec2 uv = FixUV();
    vec3 color = texture(iChannel0, uv).rgb;
    gl_FragColor = vec4(color, 1.0);

    //gl_FragColor = vec4(LookUpTalble(color), 1.0);
}
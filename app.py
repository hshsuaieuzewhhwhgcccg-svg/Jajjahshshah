import os
import sys
import subprocess
import tempfile
import gradio as gr

def generate_video(
    image, 
    second_image, 
    prompt, 
    neg_prompt, 
    task, 
    size, 
    frame_num, 
    steps, 
    solver, 
    guide_scale, 
    shift, 
    seed, 
    offload_model
):
    if image is None:
        raise gr.Error("Molimo učitajte barem prvu (glavnu) sliku za izradu videa!")
    if not prompt or not prompt.strip():
        raise gr.Error("Molimo unesite tekstualni opis pokreta (prompt)!")

    temp_dir = tempfile.mkdtemp()
    input_img_path = os.path.join(temp_dir, "input_ref.png")
    image.save(input_img_path)

    output_video_path = os.path.join(temp_dir, "output_video.mp4")

    cmd = [
        sys.executable, "generate.py",
        "--task", str(task),
        "--size", str(size),
        "--frame_num", str(frame_num),
        "--image", input_img_path,
        "--prompt", str(prompt),
        "--sample_solver", str(solver),
        "--sample_steps", str(steps),
        "--sample_guide_scale", str(guide_scale),
        "--sample_shift", str(shift),
        "--base_seed", str(int(seed)),
        "--save_file", output_video_path
    ]

    if neg_prompt and neg_prompt.strip():
        cmd.extend(["--n_prompt", str(neg_prompt)])

    if second_image is not None:
        second_img_path = os.path.join(temp_dir, "second_ref.png")
        second_image.save(second_img_path)
        cmd.extend(["--src_root_path", temp_dir])

    if offload_model:
        cmd.extend(["--offload_model", "True"])
    else:
        cmd.extend(["--offload_model", "False"])

    try:
        print("Pokretanje obrade na lokalnom GPU-u...")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("GPU Izlaz:\n", process.stdout)
    except subprocess.CalledProcessError as e:
        print("Greška pri izradi na GPU-u:\n", e.stderr)
        raise gr.Error(f"Došlo je do greške na GPU-u prilikom pokretanja Wan2.2: {e.stderr}")

    if not os.path.exists(output_video_path):
        raise gr.Error("Generiranje je završilo, ali izlazna video datoteka nije pronađena.")

    return output_video_path


css = """
footer {visibility: hidden}
.gradio-container {background-color: #0b0f19;}
"""

with gr.Blocks(title="Wan2.2 GPU Local AI Studio", css=css, theme=gr.themes.Soft(primary_hue="indigo", dark_mode=True)) as demo:
    
    gr.Markdown(
        """
        # 🎬 Wan2.2 Image-to-Video AI Studio (Lokalni GPU)
        Aplikacija izravno koristi **Vaš lokalni GPU** za izradu videa iz slika.
        """
    )

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(
                label="Prompt (Opis pokreta i radnje)", 
                placeholder="Npr. Mačka sa sunčanim naočalama sjedi na dasci za surfanje...", 
                lines=3
            )
            neg_prompt = gr.Textbox(
                label="Negativni Prompt (Što izbjegavati)", 
                placeholder="Npr. zamućeno, niska kvaliteta...", 
                lines=1
            )

            with gr.Row():
                image1 = gr.Image(type="pil", label="1. Glavna Slika (Obavezno)", sources=['upload', 'clipboard'])
                image2 = gr.Image(type="pil", label="2. Sporedna Slika / Pose / BG (Opcionalno)", sources=['upload', 'clipboard'])

            with gr.Accordion("⚙️ Napredne Postavke Wan2.2 Modela", open=True):
                with gr.Row():
                    task = gr.Dropdown(choices=["i2v-A14B", "animate-14B"], value="i2v-A14B", label="Zadatak (Model Task)")
                    size = gr.Dropdown(choices=["1280*720", "960*540", "720*1280", "540*960"], value="1280*720", label="Rezolucija (Width*Height)")
                    frame_num = gr.Dropdown(choices=[81, 49, 33], value=81, label="Broj Okvira (Frame Num - 4n+1)")

                with gr.Row():
                    steps = gr.Slider(minimum=10, maximum=50, value=40, step=1, label="Koraci uzorkovanja (Steps)")
                    solver = gr.Dropdown(choices=["unipc", "dpm++"], value="unipc", label="Sampling Solver")

                with gr.Row():
                    guide_scale = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.5, label="Guide Scale (CFG)")
                    shift = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.5, label="Shift Parameter")
                    seed = gr.Number(value=-1, label="Seed (-1 za nasumično)", precision=0)

                offload_model = gr.Checkbox(value=True, label="Offload Model to CPU (Ušteda VRAM memorije)")

            generate_btn = gr.Button("🚀 Pokreni Izradu Videa na Lokalom GPU-u", variant="primary", size="lg")

        with gr.Column(scale=5):
            video_output = gr.Video(label="Generirani Video Izlaz (MP4)", interactive=False)

    generate_btn.click(
        fn=generate_video,
        inputs=[
            image1, image2, prompt, neg_prompt, task, size, 
            frame_num, steps, solver, guide_scale, shift, seed, offload_model
        ],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", port=7860, share=False)

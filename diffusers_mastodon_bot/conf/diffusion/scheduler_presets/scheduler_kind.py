from diffusers import schedulers as schedulers


class WaifuDiffusionXLSchedulerLoader:
    def from_pretrained(self, **args):
        # scheduler args documented here:
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L98
        scheduler: schedulers.DPMSolverMultistepScheduler = schedulers.DPMSolverMultistepScheduler.from_pretrained(
            'Birchlabs/waifu-diffusion-xl-unofficial',
            subfolder='scheduler',
            algorithm_type='sde-dpmsolver++',
            solver_order=2,
            # solver_type='heun' may give a sharper image. Cheng Lu reckons midpoint is better.
            solver_type='midpoint',
            use_karras_sigmas=True,
        )
        return scheduler

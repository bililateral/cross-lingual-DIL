"""H_cons handwritten qualification using the existing bounded Linux executor."""
import step28_g_half_verify as qualification
import step28_h_consistency_run as run

if __name__ == "__main__":
    qualification.main(entry=run, handwritten_module="test_step28_h_consistency",
                       test_class_name="HConsistencyTests", consistency=True)
